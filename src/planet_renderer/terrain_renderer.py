"""GPU terrain geometry generation for the planet renderer.

This component owns the Taichi fields and kernels that turn lightweight patch
descriptors into camera-relative-ready terrain geometry.  It deliberately has
no camera, rasterizer, visibility, or compositing responsibilities; the main
renderer consumes the generated fields through an explicit backend boundary.
"""

import numpy as np
import taichi as ti

from .height import TerrainHeightModel
from .terrain_types import TerrainPatchRenderDescriptor


@ti.data_oriented
class TerrainRenderer:
    """Generate and release terrain patch geometry on the selected Taichi backend."""

    def __init__(
        self,
        height_model: TerrainHeightModel,
        max_patches: int,
        patch_resolution: int,
    ) -> None:
        if max_patches < 1:
            raise ValueError("max_patches must be positive")
        if patch_resolution < 1:
            raise ValueError("patch_resolution must be positive")

        self.max_patches = max_patches
        self.resolution = patch_resolution
        self.side = patch_resolution + 1
        self.surface_vertices = self.side * self.side
        self.vertices_per_patch = self.surface_vertices + 4 * self.side

        self.height_model = height_model
        required_gpu_methods = ("sample_height_gpu", "sample_terrain_gpu")
        if not self.height_model.supports_gpu or any(
            not hasattr(self.height_model, name) for name in required_gpu_methods
        ):
            raise ValueError("terrain height model has no complete GPU evaluator")

        self.slot_resident = ti.field(ti.i32, shape=max_patches)
        self.slot_face = ti.field(ti.i32, shape=max_patches)
        self.slot_level = ti.field(ti.i32, shape=max_patches)
        self.slot_x = ti.field(ti.i32, shape=max_patches)
        self.slot_y = ti.field(ti.i32, shape=max_patches)
        self.offset = ti.Vector.field(
            3, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        self.normal = ti.Vector.field(
            3, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        self.height_m = ti.field(ti.f32, shape=(max_patches, self.vertices_per_patch))
        self.material = ti.Vector.field(
            4, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        self.landform = ti.Vector.field(
            6, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        self.cell = ti.field(ti.i32, shape=(max_patches, self.vertices_per_patch))

        # Keep upload argument shapes stable and reuse their storage. Taichi
        # otherwise has to marshal several newly allocated, variably-sized
        # external arrays whenever the streaming set changes.
        self.upload_descriptor = ti.Vector.field(6, ti.f32, shape=max_patches)
        self._upload_host = np.zeros((max_patches, 6), np.float32)

    @ti.func
    def _direction(self, face: ti.i32, u: ti.f32, v: ti.f32):
        d = ti.Vector([1.0, v, -u])
        if face == 1:
            d = ti.Vector([-1.0, v, u])
        elif face == 2:
            d = ti.Vector([u, 1.0, -v])
        elif face == 3:
            d = ti.Vector([u, -1.0, v])
        elif face == 4:
            d = ti.Vector([u, v, 1.0])
        elif face == 5:
            d = ti.Vector([-u, v, -1.0])
        return d.normalized()

    @ti.func
    def _uv_for_vertex(self, index: ti.i32):
        edge = -1
        base = index
        if index >= self.surface_vertices:
            edge = (index - self.surface_vertices) // self.side
            base = (index - self.surface_vertices) % self.side
        ix = base % self.side
        iy = base // self.side
        if edge == 0:
            ix = base
            iy = 0
        elif edge == 1:
            ix = self.resolution
            iy = base
        elif edge == 2:
            ix = self.resolution - base
            iy = self.resolution
        elif edge == 3:
            ix = 0
            iy = self.resolution - base
        return ix, iy, edge

    @ti.kernel
    def _generate_patches(
        self,
        count: ti.i32,
    ):
        """Generate only expensive surface vertices for a batch of patches.

        Skirt vertices are derived later from already-generated edge vertices,
        so procedural FBM is evaluated exactly once per unique surface vertex.
        """

        for work in range(count * self.surface_vertices):
            upload = work // self.surface_vertices
            index = work % self.surface_vertices
            descriptor = self.upload_descriptor[upload]
            slot = ti.cast(descriptor[0], ti.i32)
            face = ti.cast(descriptor[1], ti.i32)
            level = ti.cast(descriptor[2], ti.i32)
            px = ti.cast(descriptor[3], ti.i32)
            py = ti.cast(descriptor[4], ti.i32)
            radius = descriptor[5]

            if index == 0:
                self.slot_resident[slot] = 1
                self.slot_face[slot] = face
                self.slot_level[slot] = level
                self.slot_x[slot] = px
                self.slot_y[slot] = py

            scale = ti.cast(1 << level, ti.f32)
            size = 2.0 / scale
            u0 = -1.0 + ti.cast(px, ti.f32) * size
            v0 = -1.0 + ti.cast(py, ti.f32) * size
            center = self._direction(face, u0 + size * 0.5, v0 + size * 0.5)

            ix = index % self.side
            iy = index // self.side
            global_resolution = (1 << level) * self.resolution
            global_x = px * self.resolution + ix
            global_y = py * self.resolution + iy
            u = -1.0 + 2.0 * ti.cast(global_x, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            v = -1.0 + 2.0 * ti.cast(global_y, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            d = self._direction(face, u, v)
            sample = self.height_model.sample_terrain_gpu(d)
            h = sample[0]
            offset = (d - center) * radius + d * h
            canonical = 1 << 14
            cx = ti.min(
                ti.max(ti.cast((u * 0.5 + 0.5) * canonical, ti.i32), 0),
                canonical - 1,
            )
            cy = ti.min(
                ti.max(ti.cast((v * 0.5 + 0.5) * canonical, ti.i32), 0),
                canonical - 1,
            )
            self.offset[slot, index] = offset
            self.normal[slot, index] = d
            self.height_m[slot, index] = h
            self.material[slot, index] = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.landform[slot, index] = ti.Vector(
                [sample[1], sample[2], sample[3], sample[4], sample[5], sample[6]]
            )
            self.cell[slot, index] = (face << 28) | (cy << 14) | cx

    @ti.kernel
    def _generate_normals_batch(
        self,
        count: ti.i32,
    ):
        for work in range(count * self.surface_vertices):
            upload = work // self.surface_vertices
            index = work % self.surface_vertices
            slot = ti.cast(self.upload_descriptor[upload][0], ti.i32)
            x = index % self.side
            y = index // self.side
            xl = ti.max(x - 1, 0)
            xr = ti.min(x + 1, self.resolution)
            yb = ti.max(y - 1, 0)
            yt = ti.min(y + 1, self.resolution)
            tangent_u = (
                self.offset[slot, y * self.side + xr]
                - self.offset[slot, y * self.side + xl]
            )
            tangent_v = (
                self.offset[slot, yt * self.side + x]
                - self.offset[slot, yb * self.side + x]
            )
            # Fourth-order centered differences preserve small terrain detail
            # without evaluating the procedural generator two more times per
            # vertex. Patch-edge normals use the bounded second-order form and
            # are made identical by the renderer's existing edge weld pass.
            if 1 < x < self.resolution - 1:
                tangent_u = (
                    -self.offset[slot, y * self.side + x + 2]
                    + 8.0 * self.offset[slot, y * self.side + x + 1]
                    - 8.0 * self.offset[slot, y * self.side + x - 1]
                    + self.offset[slot, y * self.side + x - 2]
                )
            if 1 < y < self.resolution - 1:
                tangent_v = (
                    -self.offset[slot, (y + 2) * self.side + x]
                    + 8.0 * self.offset[slot, (y + 1) * self.side + x]
                    - 8.0 * self.offset[slot, (y - 1) * self.side + x]
                    + self.offset[slot, (y - 2) * self.side + x]
                )
            scale = ti.cast(1 << self.slot_level[slot], ti.f32)
            size = 2.0 / scale
            u = (
                -1.0
                + (
                    ti.cast(self.slot_x[slot], ti.f32)
                    + ti.cast(x, ti.f32) / self.resolution
                )
                * size
            )
            v = (
                -1.0
                + (
                    ti.cast(self.slot_y[slot], ti.f32)
                    + ti.cast(y, ti.f32) / self.resolution
                )
                * size
            )
            radial = self._direction(self.slot_face[slot], u, v)
            normal = tangent_u.cross(tangent_v).normalized()
            if normal.dot(radial) < 0:
                normal = -normal
            self.normal[slot, index] = normal

    @ti.kernel
    def _generate_materials_batch(self, count: ti.i32):
        for work in range(count * self.surface_vertices):
            upload = work // self.surface_vertices
            index = work % self.surface_vertices
            slot = ti.cast(self.upload_descriptor[upload][0], ti.i32)
            x = index % self.side
            y = index // self.side
            scale = ti.cast(1 << self.slot_level[slot], ti.f32)
            size = 2.0 / scale
            u = (
                -1.0
                + (
                    ti.cast(self.slot_x[slot], ti.f32)
                    + ti.cast(x, ti.f32) / self.resolution
                )
                * size
            )
            v = (
                -1.0
                + (
                    ti.cast(self.slot_y[slot], ti.f32)
                    + ti.cast(y, ti.f32) / self.resolution
                )
                * size
            )
            radial = self._direction(self.slot_face[slot], u, v)
            height = self.height_m[slot, index]
            landform = self.landform[slot, index]
            land = 1.0 - landform[0]
            slope = 1.0 - ti.max(self.normal[slot, index].dot(radial), 0.0)
            snow = (
                ti.min(ti.max((height - 2700.0) / 1500.0, 0.0), 1.0)
                * land
            )
            rock = ti.min(
                ti.max(
                    slope * 3.2 + landform[2] * 0.48 + landform[5] * 0.62,
                    0.0,
                ),
                1.0,
            ) * (1.0 - snow) * land
            arid = ti.min(
                ti.max(
                    landform[3] * 0.78
                    + landform[4] * 0.38
                    + ti.max(height - 900.0, 0.0) / 5000.0,
                    0.0,
                ),
                1.0,
            ) * ti.max(1.0 - snow - rock, 0.0) * land
            fertile = ti.max(land - snow - rock - arid, 0.0)
            weights = ti.Vector([fertile, arid, rock, snow])
            weights /= ti.max(weights.sum(), 1.0e-8)
            self.material[slot, index] = weights

    @ti.kernel
    def _generate_skirts_batch(
        self,
        count: ti.i32,
    ):
        """Derive skirts from surface edges without recomputing height noise."""

        skirt_vertices = 4 * self.side
        for work in range(count * skirt_vertices):
            upload = work // skirt_vertices
            local = work % skirt_vertices
            descriptor = self.upload_descriptor[upload]
            slot = ti.cast(descriptor[0], ti.i32)
            radius = descriptor[5]
            edge = local // self.side
            base = local % self.side

            source = base
            ix = base
            iy = 0
            if edge == 1:
                source = base * self.side + self.resolution
                ix = self.resolution
                iy = base
            elif edge == 2:
                source = self.resolution * self.side + self.resolution - base
                ix = self.resolution - base
                iy = self.resolution
            elif edge == 3:
                source = (self.resolution - base) * self.side
                ix = 0
                iy = self.resolution - base

            level = self.slot_level[slot]
            px = self.slot_x[slot]
            py = self.slot_y[slot]
            face = self.slot_face[slot]
            scale = ti.cast(1 << level, ti.f32)
            size = 2.0 / scale
            global_resolution = (1 << level) * self.resolution
            global_x = px * self.resolution + ix
            global_y = py * self.resolution + iy
            u = -1.0 + 2.0 * ti.cast(global_x, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            v = -1.0 + 2.0 * ti.cast(global_y, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            radial = self._direction(face, u, v)
            cell_size = radius * size / self.resolution
            skirt_depth = ti.max(50.0, cell_size * 0.5)
            index = self.surface_vertices + local

            self.offset[slot, index] = self.offset[slot, source] - radial * skirt_depth
            self.normal[slot, index] = self.normal[slot, source]
            self.height_m[slot, index] = self.height_m[slot, source]
            self.material[slot, index] = self.material[slot, source]
            self.landform[slot, index] = self.landform[slot, source]
            self.cell[slot, index] = self.cell[slot, source]

    @ti.kernel
    def _release(self, slot: ti.i32):
        self.slot_resident[slot] = 0

    def upload_patches(
        self,
        uploads,
    ) -> None:
        """Generate a batch of patches with bounded GPU launches.

        The previous implementation launched geometry + normal kernels once per
        patch.  During movement that made the upload budget directly translate
        into many tiny dispatches and repeated FBM work for skirt vertices.
        """

        uploads = tuple(uploads)
        if not uploads:
            return
        count = len(uploads)
        if count > self.max_patches:
            raise ValueError("upload batch exceeds terrain slot capacity")
        for index, item in enumerate(uploads):
            key = item.descriptor.key
            self._upload_host[index] = (
                item.slot,
                key.face,
                key.level,
                key.x,
                key.y,
                float(np.linalg.norm(item.descriptor.anchor_global)),
            )
        self.upload_descriptor.from_numpy(self._upload_host)
        self._generate_patches(count)
        self._generate_normals_batch(count)
        self._generate_materials_batch(count)
        self._generate_skirts_batch(count)

    def release_patch(self, slot: int | None) -> None:
        """Release geometry residency; render visibility is owned by the caller."""

        if slot is not None:
            self._release(slot)
