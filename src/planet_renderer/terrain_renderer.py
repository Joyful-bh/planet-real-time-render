"""GPU terrain geometry generation for the planet renderer.

This component owns the Taichi fields and kernels that turn lightweight patch
descriptors into camera-relative-ready terrain geometry.  It deliberately has
no camera, rasterizer, visibility, or compositing responsibilities; the main
renderer consumes the generated fields through an explicit backend boundary.
"""

import numpy as np
import taichi as ti

from .height import GpuHeightProgramDescriptor, HeightProvider
from .terrain_types import TerrainPatchRenderDescriptor


@ti.data_oriented
class TerrainRenderer:
    """Generate and release terrain patch geometry on the selected Taichi backend."""

    def __init__(
        self,
        max_patches: int,
        patch_resolution: int,
        height_provider: HeightProvider | None = None,
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

        descriptor = self._height_descriptor(height_provider)
        self.seed = descriptor.seed
        self.continent_amplitude = descriptor.continent_amplitude_m
        self.mountain_amplitude = descriptor.mountain_amplitude_m

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
        self.cell = ti.field(ti.i32, shape=(max_patches, self.vertices_per_patch))

    @staticmethod
    def _height_descriptor(
        height_provider: HeightProvider | None,
    ) -> GpuHeightProgramDescriptor:
        if height_provider is None:
            return GpuHeightProgramDescriptor(
                kind=1,
                seed=7,
                continent_amplitude_m=2800.0,
                mountain_amplitude_m=4200.0,
            )

        return height_provider.gpu_descriptor()

    @ti.func
    def _hash3(self, x: ti.i32, y: ti.i32, z: ti.i32, seed: ti.i32) -> ti.f32:
        value = (
            ti.cast(x, ti.u32) * ti.u32(0x1F123BB5)
            ^ ti.cast(y, ti.u32) * ti.u32(0x05491333)
            ^ ti.cast(z, ti.u32) * ti.u32(0x72E12A4D)
            ^ ti.cast(seed, ti.u32)
        )
        value = (value ^ (value >> 15)) * ti.u32(0x2C1B3C6D)
        value = (value ^ (value >> 12)) * ti.u32(0x297A2D39)
        value = value ^ (value >> 15)
        return ti.cast(value, ti.f32) / 4294967295.0

    @ti.func
    def _noise(self, p: ti.template(), seed: ti.i32) -> ti.f32:
        base = ti.cast(ti.floor(p), ti.i32)
        f = p - ti.cast(base, ti.f32)
        w = f * f * (3.0 - 2.0 * f)
        value = 0.0
        for dz, dy, dx in ti.static(ti.ndrange(2, 2, 2)):
            value += (
                self._hash3(base.x + dx, base.y + dy, base.z + dz, seed)
                * (w.x if dx else 1.0 - w.x)
                * (w.y if dy else 1.0 - w.y)
                * (w.z if dz else 1.0 - w.z)
            )
        return value

    @ti.func
    def _fbm(self, p: ti.template(), seed: ti.i32, octaves: ti.i32) -> ti.f32:
        value = 0.0
        total = 0.0
        amplitude = 0.5
        point = p
        for octave in ti.static(range(5)):
            if octave < octaves:
                value += self._noise(point, seed + octave * 1013) * amplitude
                total += amplitude
            point = point * 2.03 + ti.Vector([7.1, -3.7, 5.3])
            amplitude *= 0.5
        return value / total

    @ti.func
    def _height(self, d: ti.template()) -> ti.f32:
        warp = (
            ti.Vector(
                [
                    self._fbm(d * 3.1 + 11.0, self.seed + 17, 3),
                    self._fbm(d * 3.1 - 7.0, self.seed + 31, 3),
                    self._fbm(d * 3.1 + 3.0, self.seed + 47, 3),
                ]
            )
            - 0.5
        )
        continent = (
            (self._fbm(d * 1.65 + warp * 0.7, self.seed, 5) - 0.5)
            * self.continent_amplitude
            * 2.0
        )
        ridge_noise = self._fbm(d * 8.0 + warp, self.seed + 211, 5)
        ridge = (1.0 - ti.abs(ridge_noise * 2.0 - 1.0)) ** 3
        land = ti.min(ti.max((continent + 700.0) / 1800.0, 0.0), 1.0)
        return ti.min(
            ti.max(continent + ridge * self.mountain_amplitude * land, -5000.0),
            8500.0,
        )

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
    def _generate_patch(
        self,
        slot: ti.i32,
        face: ti.i32,
        level: ti.i32,
        px: ti.i32,
        py: ti.i32,
        radius: ti.f32,
    ):
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

        global_resolution = (1 << level) * self.resolution
        for index in range(self.vertices_per_patch):
            ix, iy, edge = self._uv_for_vertex(index)
            global_x = px * self.resolution + ix
            global_y = py * self.resolution + iy
            u = -1.0 + 2.0 * ti.cast(global_x, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            v = -1.0 + 2.0 * ti.cast(global_y, ti.f32) / ti.cast(
                global_resolution, ti.f32
            )
            d = self._direction(face, u, v)
            h = self._height(d)
            offset = (d - center) * radius + d * h
            if edge >= 0:
                cell_size = radius * size / self.resolution
                skirt_depth = ti.max(50.0, cell_size * 0.5)
                offset -= d * skirt_depth
            snow = ti.min(ti.max((h - 2600.0) / 1800.0, 0.0), 1.0)
            sand = ti.min(ti.max(1.0 - ti.abs(h) / 500.0, 0.0), 1.0) * (1.0 - snow)
            soil = ti.max(1.0 - snow - sand, 0.0)
            weights = ti.Vector([soil, 0.0, sand, snow])
            weights /= ti.max(weights.sum(), 1e-8)
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
            self.material[slot, index] = weights
            self.cell[slot, index] = (face << 28) | (cy << 14) | cx

    @ti.kernel
    def _generate_normals(self, slot: ti.i32):
        for index in range(self.surface_vertices):
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
            normal = tangent_u.cross(tangent_v).normalized()
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
            if normal.dot(radial) < 0:
                normal = -normal
            self.normal[slot, index] = normal
        for index in range(self.surface_vertices, self.vertices_per_patch):
            base = (index - self.surface_vertices) % self.side
            edge = (index - self.surface_vertices) // self.side
            source = base
            if edge == 1:
                source = base * self.side + self.resolution
            elif edge == 2:
                source = self.resolution * self.side + self.resolution - base
            elif edge == 3:
                source = (self.resolution - base) * self.side
            self.normal[slot, index] = self.normal[slot, source]

    @ti.kernel
    def _release(self, slot: ti.i32):
        self.slot_resident[slot] = 0

    def upload_patch(
        self,
        slot: int,
        descriptor: TerrainPatchRenderDescriptor,
    ) -> None:
        """Generate one patch and its normals in a reusable GPU slot."""

        key = descriptor.key
        self._generate_patch(
            slot,
            key.face,
            key.level,
            key.x,
            key.y,
            float(np.linalg.norm(descriptor.anchor_global)),
        )
        self._generate_normals(slot)

    def release_patch(self, slot: int | None) -> None:
        """Release geometry residency; render visibility is owned by the caller."""

        if slot is not None:
            self._release(slot)
