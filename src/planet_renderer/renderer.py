"""Taichi software rasterization and frame composition for one planet.

Terrain geometry generation lives in :mod:`terrain_renderer`; this module
owns camera-relative transforms, clipping, rasterization, G-buffer writes and
display composition.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from .camera import PlanetCamera
from .height import HeightProvider
from .lighting import LightingState
from .planet import PlanetModel
from .postprocess import display_transform
from .terrain_lod import cube_face_direction
from .terrain_renderer import TerrainRenderer
from .terrain_types import PatchKey, TerrainFrame, TerrainPatchRenderDescriptor

TILE_SIZE = 16
MAX_TRIANGLES_PER_TILE = 512
MAX_CLIP_VERTICES = 8
MAX_CLIPPED_TRIANGLES = 6


@dataclass(frozen=True)
class TimingResult:
    jit_seconds: float
    average_seconds: float
    frames_per_second: float


@dataclass(frozen=True)
class RasterStats:
    patches: int
    vertices: int
    triangles: int
    tile_overflow: int


def create_cube_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("立方体球细分必须至少为 2")
    vertices = []
    triangles = []
    side = n + 1
    for face in range(6):
        base = len(vertices)
        for y in range(side):
            for x in range(side):
                vertices.append(cube_face_direction(face, x / n * 2 - 1, y / n * 2 - 1))
        for y in range(n):
            for x in range(n):
                a = base + y * side + x
                b = a + 1
                c = a + side
                d = c + 1
                triangles.extend(((a, b, c), (b, d, c)))
    return np.asarray(vertices, np.float32), np.asarray(triangles, np.int32)


def _shared_topology(n: int) -> np.ndarray:
    side = n + 1
    tri = []
    for y in range(n):
        for x in range(n):
            a = y * side + x
            b = a + 1
            c = a + side
            d = c + 1
            tri.extend(((a, b, c), (b, d, c)))
    edges = (
        tuple(range(side)),
        tuple(y * side + n for y in range(side)),
        tuple(range(n * side, n * side + side))[::-1],
        tuple(y * side for y in range(side))[::-1],
    )
    base = side * side
    for edge in edges:
        for i in range(n):
            a, b, c, d = edge[i], edge[i + 1], base + i, base + i + 1
            tri.extend(((a, c, b), (b, c, d)))
        base += side
    return np.asarray(tri, np.int32)


@ti.data_oriented
class PlanetRenderer:
    """Render terrain geometry supplied by the dedicated terrain backend."""

    def __init__(
        self,
        width: int,
        height: int,
        max_patches: int = 256,
        patch_resolution: int = 12,
        height_provider: HeightProvider | None = None,
    ):
        self.width, self.height = width, height
        self.max_patches = max_patches
        self.resolution = patch_resolution
        self.side = patch_resolution + 1
        self.surface_vertices = self.side * self.side
        self.vertices_per_patch = self.surface_vertices + 4 * self.side
        topology = _shared_topology(patch_resolution)
        self.local_triangle_count = len(topology)
        self.surface_triangle_count = patch_resolution * patch_resolution * 2
        self.tiles_x = (width + 15) // 16
        self.tiles_y = (height + 15) // 16
        self.patch_count = 0
        self.debug_view = 0
        self.terrain_renderer = TerrainRenderer(
            max_patches=max_patches,
            patch_resolution=patch_resolution,
            height_provider=height_provider,
        )
        self.local_triangles = ti.Vector.field(
            3, ti.i32, shape=self.local_triangle_count
        )
        self.local_triangles.from_numpy(topology)
        self.slot_render = ti.field(ti.i32, shape=max_patches)
        self.slot_skirt_mask = ti.field(
            ti.i32,
            shape=max_patches,
        )
        self.slot_stitch_mask = ti.field(
            ti.i32,
            shape=max_patches,
        )
        # Render-patch boundaries are welded after camera transformation.  A
        # shared theoretical cube-sphere vertex otherwise differs slightly
        # when reconstructed through two independent float32 patch anchors.
        self.max_weld_operations = max_patches * 4 * self.side
        self.max_stitch_operations = max_patches * 4 * ((patch_resolution + 1) // 2)
        self.weld_count = ti.field(ti.i32, shape=())
        self.weld_dst = ti.Vector.field(2, ti.i32, shape=self.max_weld_operations)
        self.weld_src = ti.Vector.field(2, ti.i32, shape=self.max_weld_operations)
        self.stitch_count = ti.field(ti.i32, shape=())
        self.stitch_vertex = ti.Vector.field(
            4, ti.i32, shape=self.max_stitch_operations
        )
        self.anchor_relative = ti.Vector.field(3, ti.f32, shape=max_patches)
        self.view = ti.Vector.field(
            3, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        raster_capacity = (
            max_patches * self.local_triangle_count * MAX_CLIPPED_TRIANGLES
        )
        rs = (raster_capacity, 3)
        self.rv = ti.Vector.field(3, ti.f32, shape=rs)
        self.rn = ti.Vector.field(3, ti.f32, shape=rs)
        self.rm = ti.Vector.field(4, ti.f32, shape=rs)
        self.rh = ti.field(ti.f32, shape=rs)
        self.rc = ti.field(ti.i32, shape=rs)
        self.screen = ti.Vector.field(3, ti.f32, shape=rs)
        self.source = ti.field(ti.i32, shape=raster_capacity)
        self.valid = ti.field(ti.i32, shape=raster_capacity)
        self.tile_counts = ti.field(ti.i32, shape=(self.tiles_x, self.tiles_y))
        self.tile_triangles = ti.field(
            ti.i32, shape=(self.tiles_x, self.tiles_y, MAX_TRIANGLES_PER_TILE)
        )
        self.tile_overflow = ti.field(ti.i32, shape=())
        shape = (width, height)
        self.depth = ti.field(ti.f32, shape=shape)
        self.gbuffer_position = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_normal = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_albedo = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_material_weights = ti.Vector.field(4, ti.f32, shape=shape)
        self.gbuffer_height_m = ti.field(ti.f32, shape=shape)
        self.gbuffer_surface_id = ti.field(ti.i32, shape=shape)
        self.gbuffer_surface_cell_id = ti.field(ti.i32, shape=shape)
        self.hdr = ti.Vector.field(3, ti.f32, shape=shape)
        self.display = ti.Vector.field(3, ti.f32, shape=shape)
        self._render_descriptors: tuple[TerrainPatchRenderDescriptor, ...] = ()
        self._render_slots: dict[PatchKey, int] = {}
        self._edge_signature: tuple | None = None
        self._render_signature: tuple | None = None

    def _surface_edge_indices(self, edge: int) -> tuple[int, ...]:
        n = self.resolution
        s = self.side
        if edge == 0:
            return tuple(x for x in range(s))
        if edge == 1:
            return tuple(y * s + n for y in range(s))
        if edge == 2:
            return tuple(n * s + x for x in range(s))
        return tuple(y * s for y in range(s))

    def _vertex_direction_key(self, key: PatchKey, index: int) -> tuple[int, int, int]:
        x = index % self.side
        y = index // self.side
        total = (1 << key.level) * self.resolution
        u = -total + 2 * (key.x * self.resolution + x)
        v = -total + 2 * (key.y * self.resolution + y)
        mappings = (
            (total, v, -u),
            (-total, v, u),
            (u, total, -v),
            (u, -total, v),
            (u, v, total),
            (-u, v, -total),
        )
        raw = mappings[key.face]
        divisor = math.gcd(math.gcd(abs(raw[0]), abs(raw[1])), abs(raw[2]))
        # Normalizing the integer cube vector by its gcd gives an exact,
        # allocation-free identity shared by levels and cube faces.
        return raw[0] // divisor, raw[1] // divisor, raw[2] // divisor

    def _build_edge_operations(
        self,
        descriptors: list[TerrainPatchRenderDescriptor],
        slots: dict[PatchKey, int | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
        stitch: list[tuple[int, int, int, int]] = []
        for descriptor in descriptors:
            slot = slots.get(descriptor.key)
            if slot is None:
                continue
            for edge in range(4):
                indices = self._surface_edge_indices(edge)
                for index in indices:
                    groups.setdefault(
                        self._vertex_direction_key(descriptor.key, index), set()
                    ).add((int(slot), index))
                if descriptor.stitch_mask & (1 << edge):
                    for position in range(1, self.resolution, 2):
                        stitch.append(
                            (
                                int(slot),
                                indices[position],
                                indices[position - 1],
                                indices[position + 1],
                            )
                        )
        destinations: list[tuple[int, int]] = []
        sources: list[tuple[int, int]] = []
        for vertices in groups.values():
            if len(vertices) < 2:
                continue
            owner = min(vertices)
            for vertex in sorted(vertices):
                if vertex != owner:
                    destinations.append(vertex)
                    sources.append(owner)
        if (
            len(destinations) > self.max_weld_operations
            or len(stitch) > self.max_stitch_operations
        ):
            raise RuntimeError("Patch edge operation capacity exceeded")
        return (
            np.asarray(destinations, np.int32).reshape((-1, 2)),
            np.asarray(sources, np.int32).reshape((-1, 2)),
            np.asarray(stitch, np.int32).reshape((-1, 4)),
        )

    @ti.kernel
    def _set_edge_operations(
        self,
        dst: ti.types.ndarray(dtype=ti.i32, ndim=2),
        src: ti.types.ndarray(dtype=ti.i32, ndim=2),
        stitch: ti.types.ndarray(dtype=ti.i32, ndim=2),
    ):
        self.weld_count[None] = dst.shape[0]
        self.stitch_count[None] = stitch.shape[0]
        for i in range(dst.shape[0]):
            self.weld_dst[i] = ti.Vector([dst[i, 0], dst[i, 1]])
            self.weld_src[i] = ti.Vector([src[i, 0], src[i, 1]])
        for i in range(stitch.shape[0]):
            self.stitch_vertex[i] = ti.Vector(
                [stitch[i, 0], stitch[i, 1], stitch[i, 2], stitch[i, 3]]
            )

    def apply_terrain_frame(self, frame: TerrainFrame) -> None:
        """Apply renderer operations emitted by :class:`CubeSphereTerrain`.

        Terrain selection owns patch residency; this method is the explicit
        runtime boundary that turns those data-only operations into GPU work.
        Releases are submitted before uploads so a slot can be reused safely.
        """

        for release in frame.releases:
            self.terrain_renderer.release_patch(release.slot)
            self._release_render_slot(release.slot)
        for upload in frame.uploads:
            self.terrain_renderer.upload_patch(upload.slot, upload.descriptor)
        self._set_render_patches(list(frame.render), dict(frame.render_slots))

    @ti.kernel
    def _release_render_slot(self, slot: ti.i32):
        self.slot_render[slot] = 0

    def _set_render_patches(
        self,
        descriptors: list[TerrainPatchRenderDescriptor],
        slots: dict[PatchKey, int | None],
    ) -> None:
        self._render_descriptors = tuple(descriptors)
        self._render_slots = {k: int(v) for k, v in slots.items() if v is not None}
        render_signature = tuple(
            sorted(
                (
                    descriptor.key,
                    int(slots[descriptor.key]),
                    descriptor.skirt_mask,
                    descriptor.stitch_mask,
                )
                for descriptor in descriptors
                if slots.get(descriptor.key) is not None
            )
        )
        if render_signature == self._render_signature:
            self.patch_count = len(render_signature)
            return
        active = np.zeros(self.max_patches, np.int32)
        skirt_masks = np.zeros(
            self.max_patches,
            np.int32,
        )
        stitch_masks = np.zeros(
            self.max_patches,
            np.int32,
        )
        for descriptor in descriptors:
            slot = slots.get(descriptor.key)
            if slot is not None:
                active[slot] = 1
                skirt_masks[slot] = descriptor.skirt_mask
                stitch_masks[slot] = descriptor.stitch_mask
        self._set_render(
            active,
            skirt_masks,
            stitch_masks,
        )
        edge_signature = tuple(
            sorted(
                (descriptor.key, int(slots[descriptor.key]), descriptor.stitch_mask)
                for descriptor in descriptors
                if slots.get(descriptor.key) is not None
            )
        )
        if edge_signature != self._edge_signature:
            weld_dst, weld_src, stitch = self._build_edge_operations(descriptors, slots)
            self._set_edge_operations(weld_dst, weld_src, stitch)
            self._edge_signature = edge_signature
        self._render_signature = render_signature
        self.patch_count = int(active.sum())

    @ti.kernel
    def _set_render(
        self,
        active: ti.types.ndarray(dtype=ti.i32, ndim=1),
        skirt_masks: ti.types.ndarray(
            dtype=ti.i32,
            ndim=1,
        ),
        stitch_masks: ti.types.ndarray(
            dtype=ti.i32,
            ndim=1,
        ),
    ):
        for i in range(self.max_patches):
            self.slot_render[i] = active[i]
            self.slot_skirt_mask[i] = skirt_masks[i]
            self.slot_stitch_mask[i] = stitch_masks[i]

    @ti.kernel
    def _set_anchors(self, a: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i in range(self.max_patches):
            self.anchor_relative[i] = ti.Vector([a[i, 0], a[i, 1], a[i, 2]])

    @ti.kernel
    def _clear(self):
        self.tile_overflow[None] = 0
        for q in ti.grouped(self.tile_counts):
            self.tile_counts[q] = 0
        for q in ti.grouped(self.depth):
            self.depth[q] = 1e30
            self.gbuffer_surface_id[q] = -1
            self.gbuffer_surface_cell_id[q] = -1

    @ti.kernel
    def _transform(
        self,
        e: ti.types.vector(3, ti.f32),
        u: ti.types.vector(3, ti.f32),
        n: ti.types.vector(3, ti.f32),
        r: ti.types.vector(3, ti.f32),
        vu: ti.types.vector(3, ti.f32),
        f: ti.types.vector(3, ti.f32),
    ):
        for slot, index in self.view:
            if self.slot_render[slot]:
                g = (
                    self.anchor_relative[slot]
                    + self.terrain_renderer.offset[slot, index]
                )
                local = ti.Vector([g.dot(e), g.dot(u), g.dot(n)])
                self.view[slot, index] = ti.Vector(
                    [local.dot(r), local.dot(vu), local.dot(f)]
                )

    @ti.kernel
    def _weld_edges(self):
        for operation in range(self.weld_count[None]):
            dst = self.weld_dst[operation]
            src = self.weld_src[operation]
            self.view[dst.x, dst.y] = self.view[src.x, src.y]
            self.terrain_renderer.normal[dst.x, dst.y] = self.terrain_renderer.normal[
                src.x, src.y
            ]
            self.terrain_renderer.height_m[dst.x, dst.y] = (
                self.terrain_renderer.height_m[src.x, src.y]
            )
            self.terrain_renderer.material[dst.x, dst.y] = (
                self.terrain_renderer.material[src.x, src.y]
            )
            self.terrain_renderer.cell[dst.x, dst.y] = self.terrain_renderer.cell[
                src.x, src.y
            ]

    @ti.kernel
    def _stitch_lod_edges(self):
        for operation in range(self.stitch_count[None]):
            item = self.stitch_vertex[operation]
            slot = item.x
            vertex = item.y
            a = item.z
            b = item.w
            self.view[slot, vertex] = (self.view[slot, a] + self.view[slot, b]) * 0.5
            self.terrain_renderer.normal[slot, vertex] = (
                self.terrain_renderer.normal[slot, a]
                + self.terrain_renderer.normal[slot, b]
            ).normalized()
            self.terrain_renderer.height_m[slot, vertex] = (
                self.terrain_renderer.height_m[slot, a]
                + self.terrain_renderer.height_m[slot, b]
            ) * 0.5
            self.terrain_renderer.material[slot, vertex] = (
                self.terrain_renderer.material[slot, a]
                + self.terrain_renderer.material[slot, b]
            ) * 0.5

    @ti.func
    def _emit(
        self,
        s: ti.i32,
        src: ti.i32,
        v: ti.template(),
        n: ti.template(),
        m: ti.template(),
        h: ti.template(),
        c: ti.template(),
        tf: ti.f32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        for k in ti.static(range(3)):
            p = v[k, :]
            self.rv[s, k] = p
            self.rn[s, k] = n[k, :]
            self.rm[s, k] = m[k, :]
            self.rh[s, k] = h[k]
            self.rc[s, k] = c[k]
            projected = ti.Vector(
                [
                    (p.x / (p.z * tf * aspect) * 0.5 + 0.5) * self.width,
                    (p.y / (p.z * tf) * 0.5 + 0.5) * self.height,
                ]
            )
            projected = ti.floor(projected * 256.0 + 0.5) / 256.0
            self.screen[s, k] = ti.Vector([projected.x, projected.y, p.z])
        self.source[s] = src
        self.valid[s] = 1

    @ti.func
    def _clip_distance(
        self,
        p: ti.template(),
        plane: ti.i32,
        near: ti.f32,
        tan_x: ti.f32,
        tan_y: ti.f32,
    ) -> ti.f32:
        distance = p.z - near
        if plane == 1:
            distance = p.x + p.z * tan_x
        elif plane == 2:
            distance = -p.x + p.z * tan_x
        elif plane == 3:
            distance = p.y + p.z * tan_y
        elif plane == 4:
            distance = -p.y + p.z * tan_y
        return distance

    @ti.kernel
    def _clip(self, near: ti.f32, tf: ti.f32):
        aspect = ti.cast(self.width, ti.f32) / self.height
        tan_x = tf * aspect
        tan_y = tf
        for source_id in range(self.max_patches * self.local_triangle_count):
            slot = source_id // self.local_triangle_count
            local_id = source_id % self.local_triangle_count
            s = source_id * MAX_CLIPPED_TRIANGLES
            for output in range(MAX_CLIPPED_TRIANGLES):
                self.valid[s + output] = 0
            skirt_edge = (local_id - self.surface_triangle_count) // (
                self.resolution * 2
            )
            enabled = local_id < self.surface_triangle_count or (
                skirt_edge >= 0
                and (self.slot_skirt_mask[slot] & (1 << skirt_edge)) != 0
            )
            if self.slot_render[slot] and enabled:
                ids = self.local_triangles[local_id]
                positions = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 3)
                normals = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 3)
                materials = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 4)
                heights = ti.Vector.zero(ti.f32, MAX_CLIP_VERTICES * 2)
                cells = ti.Vector.zero(ti.i32, MAX_CLIP_VERTICES * 2)
                for vertex in range(3):
                    index = ids[vertex]
                    positions[vertex, :] = self.view[slot, index]
                    normals[vertex, :] = self.terrain_renderer.normal[slot, index]
                    materials[vertex, :] = self.terrain_renderer.material[slot, index]
                    heights[vertex] = self.terrain_renderer.height_m[slot, index]
                    cells[vertex] = self.terrain_renderer.cell[slot, index]
                count = 3
                read_buffer = 0
                for plane in range(5):
                    write_buffer = 1 - read_buffer
                    output_count = 0
                    for vertex in range(MAX_CLIP_VERTICES):
                        if vertex < count:
                            previous = (vertex + count - 1) % count
                            current_index = read_buffer * MAX_CLIP_VERTICES + vertex
                            previous_index = read_buffer * MAX_CLIP_VERTICES + previous
                            current_position = positions[current_index, :]
                            previous_position = positions[previous_index, :]
                            current_distance = self._clip_distance(
                                current_position, plane, near, tan_x, tan_y
                            )
                            previous_distance = self._clip_distance(
                                previous_position, plane, near, tan_x, tan_y
                            )
                            current_inside = current_distance >= 0.0
                            previous_inside = previous_distance >= 0.0
                            if (
                                current_inside != previous_inside
                                and output_count < MAX_CLIP_VERTICES
                            ):
                                denominator = previous_distance - current_distance
                                t = (
                                    previous_distance / denominator
                                    if ti.abs(denominator) > 1e-20
                                    else 0.0
                                )
                                destination = (
                                    write_buffer * MAX_CLIP_VERTICES + output_count
                                )
                                positions[destination, :] = (
                                    previous_position
                                    + (current_position - previous_position) * t
                                )
                                normals[destination, :] = (
                                    normals[previous_index, :]
                                    + (
                                        normals[current_index, :]
                                        - normals[previous_index, :]
                                    )
                                    * t
                                ).normalized()
                                materials[destination, :] = (
                                    materials[previous_index, :]
                                    + (
                                        materials[current_index, :]
                                        - materials[previous_index, :]
                                    )
                                    * t
                                )
                                heights[destination] = (
                                    heights[previous_index]
                                    + (heights[current_index] - heights[previous_index])
                                    * t
                                )
                                cells[destination] = (
                                    cells[current_index]
                                    if current_inside
                                    else cells[previous_index]
                                )
                                output_count += 1
                            if current_inside and output_count < MAX_CLIP_VERTICES:
                                destination = (
                                    write_buffer * MAX_CLIP_VERTICES + output_count
                                )
                                positions[destination, :] = current_position
                                normals[destination, :] = normals[current_index, :]
                                materials[destination, :] = materials[current_index, :]
                                heights[destination] = heights[current_index]
                                cells[destination] = cells[current_index]
                                output_count += 1
                    count = output_count
                    read_buffer = write_buffer
                for triangle in range(MAX_CLIPPED_TRIANGLES):
                    if triangle < count - 2:
                        a = read_buffer * MAX_CLIP_VERTICES
                        b = a + triangle + 1
                        c = a + triangle + 2
                        self._emit(
                            s + triangle,
                            source_id,
                            ti.Matrix.rows(
                                [positions[a, :], positions[b, :], positions[c, :]]
                            ),
                            ti.Matrix.rows(
                                [normals[a, :], normals[b, :], normals[c, :]]
                            ),
                            ti.Matrix.rows(
                                [materials[a, :], materials[b, :], materials[c, :]]
                            ),
                            ti.Vector([heights[a], heights[b], heights[c]]),
                            ti.Vector([cells[a], cells[b], cells[c]]),
                            tf,
                        )

    @ti.kernel
    def _bin(self):
        for q in self.valid:
            if self.valid[q]:
                source_id = self.source[q]
                local_id = source_id % self.local_triangle_count
                is_skirt = local_id >= self.surface_triangle_count
                front_facing = (self.rv[q, 1] - self.rv[q, 0]).cross(
                    self.rv[q, 2] - self.rv[q, 0]
                ).dot(-self.rv[q, 0]) < 0.0
                if is_skirt or front_facing:
                    a, b, c = self.screen[q, 0], self.screen[q, 1], self.screen[q, 2]
                    x0 = ti.max(ti.cast(ti.floor(ti.min(a.x, b.x, c.x)), ti.i32), 0)
                    x1 = ti.min(
                        ti.cast(ti.ceil(ti.max(a.x, b.x, c.x)), ti.i32), self.width - 1
                    )
                    y0 = ti.max(ti.cast(ti.floor(ti.min(a.y, b.y, c.y)), ti.i32), 0)
                    y1 = ti.min(
                        ti.cast(ti.ceil(ti.max(a.y, b.y, c.y)), ti.i32), self.height - 1
                    )
                    if x0 <= x1 and y0 <= y1:
                        for x, y in ti.ndrange(
                            (x0 // 16, x1 // 16 + 1), (y0 // 16, y1 // 16 + 1)
                        ):
                            k = ti.atomic_add(self.tile_counts[x, y], 1)
                            if k < MAX_TRIANGLES_PER_TILE:
                                self.tile_triangles[x, y, k] = q
                            else:
                                ti.atomic_add(self.tile_overflow[None], 1)

    @ti.func
    def _edge(self, a: ti.template(), b: ti.template(), p: ti.template()) -> ti.f32:
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)

    @ti.func
    def _top_left(
        self, a: ti.template(), b: ti.template(), orientation: ti.f32
    ) -> ti.i32:
        dx = (b.x - a.x) * orientation
        dy = (b.y - a.y) * orientation
        return ti.cast((dy > 0.0) or (dy == 0.0 and dx < 0.0), ti.i32)

    @ti.func
    def _height_band(self, h: ti.f32) -> ti.types.vector(3, ti.f32):
        # Height debug palette in metres.  The broad, recognisable bands make
        # subtle terrain relief readable from orbit while the short blends
        # avoid unstable one-pixel contours near a band boundary.
        deep_water = ti.Vector([0.015, 0.12, 0.34])
        shallow_water = ti.Vector([0.02, 0.63, 0.82])
        beach = ti.Vector([0.82, 0.78, 0.48])
        lowland = ti.Vector([0.16, 0.58, 0.22])
        upland = ti.Vector([0.48, 0.42, 0.16])
        rock = ti.Vector([0.43, 0.39, 0.35])
        snow = ti.Vector([0.94, 0.96, 0.98])
        color = deep_water
        if h < -250.0:
            t = ti.min(ti.max((h + 5000.0) / 4750.0, 0.0), 1.0)
            color = deep_water * (1.0 - t) + shallow_water * t
        elif h < 0.0:
            color = shallow_water
        elif h < 120.0:
            t = ti.min(ti.max(h / 120.0, 0.0), 1.0)
            color = beach * (1.0 - t) + lowland * t
        elif h < 1400.0:
            t = ti.min(ti.max((h - 120.0) / 1280.0, 0.0), 1.0)
            color = lowland * (1.0 - t) + upland * t
        elif h < 3000.0:
            t = ti.min(ti.max((h - 1400.0) / 1600.0, 0.0), 1.0)
            color = upland * (1.0 - t) + rock * t
        else:
            t = ti.min(ti.max((h - 3000.0) / 1400.0, 0.0), 1.0)
            color = rock * (1.0 - t) + snow * t
        return color

    @ti.func
    def _debug_color(self, h: ti.f32, slot: ti.i32, mode: ti.i32):
        color = self._height_band(h)
        if mode == 1:
            hue = ti.cast(self.terrain_renderer.slot_level[slot] % 6, ti.f32) / 6.0
            color = ti.Vector(
                [
                    ti.abs(hue * 6.0 - 3.0) - 1.0,
                    2.0 - ti.abs(hue * 6.0 - 2.0),
                    2.0 - ti.abs(hue * 6.0 - 4.0),
                ]
            )
            color = ti.min(ti.max(color, 0.0), 1.0)
        elif mode == 2:
            value = ti.cast((slot * 1103515245 + 12345) & 255, ti.f32) / 255.0
            color = ti.Vector(
                [
                    value,
                    ti.math.fract(value * 0.73 + 0.21),
                    ti.math.fract(value * 0.37 + 0.61),
                ]
            )
        return color

    @ti.kernel
    def _raster(self, mode: ti.i32):
        for x, y in self.depth:
            p = ti.Vector([ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5])
            best = 1e30
            src = -1
            cell = -1
            bp = ti.Vector.zero(ti.f32, 3)
            bn = ti.Vector.zero(ti.f32, 3)
            bm = ti.Vector.zero(ti.f32, 4)
            bh = 0.0
            for k in range(
                ti.min(self.tile_counts[x // 16, y // 16], MAX_TRIANGLES_PER_TILE)
            ):
                q = self.tile_triangles[x // 16, y // 16, k]
                a, b, c = self.screen[q, 0], self.screen[q, 1], self.screen[q, 2]
                area = self._edge(a.xy, b.xy, c.xy)
                sg = 1.0 if area > 0 else -1.0
                e = ti.Vector(
                    [
                        self._edge(b.xy, c.xy, p) * sg,
                        self._edge(c.xy, a.xy, p) * sg,
                        self._edge(a.xy, b.xy, p) * sg,
                    ]
                )
                inside = (
                    (e.x > 0.0 or (e.x == 0.0 and self._top_left(b.xy, c.xy, sg) != 0))
                    and (
                        e.y > 0.0
                        or (e.y == 0.0 and self._top_left(c.xy, a.xy, sg) != 0)
                    )
                    and (
                        e.z > 0.0
                        or (e.z == 0.0 and self._top_left(a.xy, b.xy, sg) != 0)
                    )
                )
                if ti.abs(area) > 1e-8 and inside:
                    l = e / ti.abs(area)
                    z = 1.0 / ti.max(l.x / a.z + l.y / b.z + l.z / c.z, 1e-20)
                    if z < best:
                        w = ti.Vector([l.x * z / a.z, l.y * z / b.z, l.z * z / c.z])
                        bp = (
                            self.rv[q, 0] * w.x
                            + self.rv[q, 1] * w.y
                            + self.rv[q, 2] * w.z
                        )
                        bn = (
                            self.rn[q, 0] * w.x
                            + self.rn[q, 1] * w.y
                            + self.rn[q, 2] * w.z
                        ).normalized()
                        bm = (
                            self.rm[q, 0] * w.x
                            + self.rm[q, 1] * w.y
                            + self.rm[q, 2] * w.z
                        )
                        bh = (
                            self.rh[q, 0] * w.x
                            + self.rh[q, 1] * w.y
                            + self.rh[q, 2] * w.z
                        )
                        vi = 0
                        if w.y > w.x:
                            vi = 1
                        if w.z > w[vi]:
                            vi = 2
                        cell = self.rc[q, vi]
                        best = z
                        src = self.source[q]
            if src >= 0:
                slot = src // self.local_triangle_count
                self.depth[x, y] = best
                self.gbuffer_surface_id[x, y] = src
                self.gbuffer_surface_cell_id[x, y] = cell
                self.gbuffer_position[x, y] = bp
                self.gbuffer_normal[x, y] = bn
                self.gbuffer_material_weights[x, y] = bm
                self.gbuffer_height_m[x, y] = bh
                self.gbuffer_albedo[x, y] = self._debug_color(bh, slot, mode)

    @ti.kernel
    def _shade(
        self,
        sun_local: ti.types.vector(3, ti.f32),
        disk: ti.types.vector(3, ti.f32),
        disk_cos: ti.f32,
        r: ti.types.vector(3, ti.f32),
        vu: ti.types.vector(3, ti.f32),
        f: ti.types.vector(3, ti.f32),
        tf: ti.f32,
        ev: ti.f32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        for q in ti.grouped(self.hdr):
            color = ti.Vector([0.00015, 0.0002, 0.00035])
            if self.gbuffer_surface_id[q] >= 0:
                color = self.gbuffer_albedo[q]
            else:
                sx = ((ti.cast(q.x, ti.f32) + 0.5) / self.width * 2.0 - 1.0) * aspect
                sy = (ti.cast(q.y, ti.f32) + 0.5) / self.height * 2.0 - 1.0
                ray = (r * sx + vu * sy + f / tf).normalized()
                if ray.dot(sun_local) >= disk_cos:
                    color += disk
            self.hdr[q] = ti.max(color, 0.0)
            self.display[q] = display_transform(color, ev)

    def render(
        self,
        planet: PlanetModel,
        camera: PlanetCamera,
        lighting: LightingState,
        surface_albedo: tuple[float, float, float],
        exposure_ev: float,
    ) -> None:
        self.planet_radius = planet.radius_m
        anchors = np.zeros((self.max_patches, 3), np.float32)
        for descriptor in self._render_descriptors:
            slot = self._render_slots.get(descriptor.key)
            if slot is not None:
                anchors[slot] = (
                    descriptor.anchor_global - camera.position_global
                ).astype(np.float32)
        self._set_anchors(anchors)
        frame = camera.frame(planet)
        r, vu, f = camera.view_basis_local()
        tf = math.tan(math.radians(camera.vertical_fov_degrees) / 2)
        near = 0.1
        sun_local = frame.global_to_local_direction(lighting.sun_direction_global)
        self._clear()
        self._transform(
            tuple(frame.east),
            tuple(frame.up),
            tuple(frame.north),
            tuple(r),
            tuple(vu),
            tuple(f),
        )
        self._weld_edges()
        self._stitch_lod_edges()
        self._clip(near, tf)
        self._bin()
        self._raster(self.debug_view)
        self._shade(
            tuple(sun_local),
            lighting.sun_disk_radiance,
            math.cos(math.radians(lighting.sun_angular_radius_degrees)),
            tuple(r),
            tuple(vu),
            tuple(f),
            tf,
            exposure_ev,
        )

    @property
    def vertex_count(self) -> int:
        return self.patch_count * self.vertices_per_patch

    @property
    def triangle_count(self) -> int:
        return self.patch_count * self.local_triangle_count

    def benchmark(self, call, repeats: int) -> TimingResult:
        s = time.perf_counter()
        call()
        ti.sync()
        jit = time.perf_counter() - s
        n = max(repeats, 1)
        s = time.perf_counter()
        for _ in range(n):
            call()
        ti.sync()
        avg = (time.perf_counter() - s) / n
        return TimingResult(jit, avg, 1 / max(avg, 1e-9))

    def stats(self) -> RasterStats:
        return RasterStats(
            self.patch_count,
            self.vertex_count,
            self.triangle_count,
            int(self.tile_overflow[None]),
        )

    def display_numpy(self) -> np.ndarray:
        return self.display.to_numpy()

    def hdr_numpy(self) -> np.ndarray:
        return self.hdr.to_numpy()
