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

from .atmosphere import (
    AtmosphereConfig,
    AtmosphereDiagnosticView,
    AtmosphereRenderer,
)
from .camera import PlanetCamera
from .height import TerrainHeightModel
from .lighting import LightingState
from .ocean import OCEAN_SURFACE_ID, OceanConfig, OceanRenderer
from .planet import PlanetModel
from .postprocess import PostprocessConfig, PostProcessor
from .space import SpaceConfig, SpaceRenderer
from .terrain_lod import cube_face_direction
from .terrain_renderer import TerrainRenderer
from .terrain_types import PatchKey, TerrainFrame, TerrainPatchRenderDescriptor

TILE_SIZE = 8
MAX_TRIANGLES_PER_TILE = 512
MAX_CLIP_VERTICES = 8
MAX_CLIPPED_TRIANGLES = 6
DEPTH_KEY_Q_BITS = 32
DEPTH_KEY_Q_MASK = (1 << DEPTH_KEY_Q_BITS) - 1
DEPTH_KEY_SENTINEL = (1 << 64) - 1
TRIANGLE_RASTER_THRESHOLD = 256
RASTER_PROBE_INTERVAL = 60


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
        height_model: TerrainHeightModel,
        atmosphere_config: AtmosphereConfig,
        postprocess_config: PostprocessConfig,
        max_patches: int = 256,
        patch_resolution: int = 12,
        ocean_config: OceanConfig | None = None,
        space_config: SpaceConfig | None = None,
    ):
        self.width, self.height = width, height
        self.max_patches = max_patches
        self.resolution = patch_resolution
        self.side = patch_resolution + 1
        self._edge_indices = tuple(
            self._surface_edge_indices(edge) for edge in range(4)
        )
        self.surface_vertices = self.side * self.side
        self.vertices_per_patch = self.surface_vertices + 4 * self.side
        topology = _shared_topology(patch_resolution)
        self.local_triangle_count = len(topology)
        self.surface_triangle_count = patch_resolution * patch_resolution * 2
        self.tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
        self.tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
        self.patch_count = 0
        self.debug_view = 0
        self.atmosphere_diagnostic_view = AtmosphereDiagnosticView.COMPOSITE
        self.terrain_renderer = TerrainRenderer(
            height_model=height_model,
            max_patches=max_patches,
            patch_resolution=patch_resolution,
        )
        self.atmosphere_renderer = AtmosphereRenderer(
            width,
            height,
            atmosphere_config,
        )
        self.ocean_renderer = OceanRenderer(
            width,
            height,
            ocean_config or OceanConfig(),
        )
        self.space_renderer = SpaceRenderer(
            width,
            height,
            space_config or SpaceConfig(),
        )
        self.postprocessor = PostProcessor(width, height, postprocess_config)
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
        # Render Set slots are kept in a dense device-side list.  The slot
        # allocator is intentionally sparse because it is shared with the
        # terrain cache, so kernels must never use the slot number as a loop
        # bound.
        self.active_slot_ids = ti.field(ti.i32, shape=max_patches)
        self.active_slot_count = ti.field(ti.i32, shape=())
        # Render-patch boundaries are welded after camera transformation.  A
        # shared theoretical cube-sphere vertex otherwise differs slightly
        # when reconstructed through two independent float32 patch anchors.
        self.max_weld_operations = max_patches * 4 * self.side
        self.max_stitch_operations = max_patches * 4 * ((patch_resolution + 1) // 2)
        self.weld_count = ti.field(ti.i32, shape=())
        self.stitch_count = ti.field(ti.i32, shape=())
        self.edge_operations = ti.Vector.field(
            4, ti.i32, shape=self.max_stitch_operations
            + self.max_weld_operations
        )
        self._edge_vertex_cache: dict[PatchKey, np.ndarray] = {}
        self.anchor_relative = ti.Vector.field(3, ti.f32, shape=max_patches)
        self.view = ti.Vector.field(
            3, ti.f32, shape=(max_patches, self.vertices_per_patch)
        )
        # Edge welding and mixed-LOD stitching are frame-local render
        # operations.  Never apply them to TerrainRenderer's resident source
        # data: that data survives render-set changes and must remain exactly
        # as generated for the patch.  These fields form the mutable attribute
        # stream consumed by clipping and rasterization for the current frame.
        frame_shape = (max_patches, self.vertices_per_patch)
        self.frame_normal = ti.Vector.field(3, ti.f32, shape=frame_shape)
        self.frame_material = ti.Vector.field(4, ti.f32, shape=frame_shape)
        self.frame_height_m = ti.field(ti.f32, shape=frame_shape)
        self.frame_cell = ti.field(ti.i32, shape=frame_shape)
        raster_capacity = (
            max_patches * self.local_triangle_count * MAX_CLIPPED_TRIANGLES
        )
        self.raster_capacity = raster_capacity
        rs = (raster_capacity, 3)
        self.rv = ti.Vector.field(3, ti.f32, shape=rs)
        self.rn = ti.Vector.field(3, ti.f32, shape=rs)
        self.rm = ti.Vector.field(4, ti.f32, shape=rs)
        self.rh = ti.field(ti.f32, shape=rs)
        self.rc = ti.field(ti.i32, shape=rs)
        self.screen = ti.Vector.field(3, ti.f32, shape=rs)
        self.source = ti.field(ti.i32, shape=raster_capacity)
        # Per-triangle screen-space setup.  Edge equations are computed once
        # when the triangle is emitted instead of once per pixel/candidate.
        self.edge_equation = ti.Vector.field(
            3, ti.f32, shape=(raster_capacity, 3)
        )
        self.inv_area = ti.field(ti.f32, shape=raster_capacity)
        self.edge_top_left = ti.Vector.field(3, ti.i32, shape=raster_capacity)
        self.clipped_count = ti.field(ti.i32, shape=())
        self.tile_counts = ti.field(ti.i32, shape=(self.tiles_x, self.tiles_y))
        self.tile_triangles = ti.field(
            ti.i32, shape=(self.tiles_x, self.tiles_y, MAX_TRIANGLES_PER_TILE)
        )
        self.tile_overflow = ti.field(ti.i32, shape=())
        self.max_tile_candidates = ti.field(ti.i32, shape=())
        # The depth pass stores a sortable (positive-float-depth, triangle-id)
        # key.  This lets all triangle fragments update the depth buffer with
        # one atomic operation, while the resolve pass interpolates the
        # winning triangle exactly once per pixel.
        self.depth_key = ti.field(ti.u64, shape=(width, height))
        shape = (width, height)
        self.depth = ti.field(ti.f32, shape=shape)
        self.gbuffer_position = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_normal = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_albedo = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_material_weights = ti.Vector.field(4, ti.f32, shape=shape)
        self.gbuffer_height_m = ti.field(ti.f32, shape=shape)
        self.gbuffer_water_depth_m = ti.field(ti.f32, shape=shape)
        self.gbuffer_seabed_albedo = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_seabed_normal = ti.Vector.field(3, ti.f32, shape=shape)
        self.gbuffer_ocean_slope_variance = ti.field(ti.f32, shape=shape)
        self.gbuffer_surface_id = ti.field(ti.i32, shape=shape)
        self.gbuffer_surface_cell_id = ti.field(ti.i32, shape=shape)
        # The ocean pass replaces the visible G-buffer sample.  Refraction
        # still needs the opaque terrain that existed before that replacement,
        # so retain a compact pre-water snapshot for screen-space lookup.
        self.terrain_position = ti.Vector.field(3, ti.f32, shape=shape)
        self.terrain_normal = ti.Vector.field(3, ti.f32, shape=shape)
        self.terrain_albedo = ti.Vector.field(3, ti.f32, shape=shape)
        self.terrain_surface_id = ti.field(ti.i32, shape=shape)
        self.surface_hdr = ti.Vector.field(3, ti.f32, shape=shape)
        self.hdr = ti.Vector.field(3, ti.f32, shape=shape)
        self.display = ti.Vector.field(3, ti.f32, shape=shape)
        self._render_descriptors: tuple[TerrainPatchRenderDescriptor, ...] = ()
        self._render_slots: dict[PatchKey, int] = {}
        self._edge_signature: tuple | None = None
        self._render_signature: tuple | None = None
        self._raster_probe_frame = 0
        self._use_triangle_raster = False
        self.last_max_tile_candidates = 0
        self.last_tile_overflow = 0
        self._active_host = np.zeros(max_patches, np.int32)
        self._skirt_masks_host = np.zeros(max_patches, np.int32)
        self._stitch_masks_host = np.zeros(max_patches, np.int32)
        self._active_ids_host = np.zeros(max_patches, np.int32)
        self._anchors_host = np.zeros((max_patches, 3), np.float32)

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

    def _patch_edge_vertices(
        self,
        key: PatchKey,
    ) -> np.ndarray:
        cached = self._edge_vertex_cache.get(key)
        if cached is not None:
            return cached
        entries = np.asarray(
            [(*self._vertex_direction_key(key, index), index)
            for indices in self._edge_indices
            for index in indices],
            dtype=np.int64,
        )
        entries = np.unique(entries, axis=0)
        self._edge_vertex_cache[key] = entries
        return entries

    def _build_edge_operations(
        self,
        descriptors: list[TerrainPatchRenderDescriptor],
        slots: dict[PatchKey, int | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_rows: list[tuple[np.ndarray, int]] = []
        row_count = 0
        stitch: list[tuple[int, int, int, int]] = []
        for descriptor in descriptors:
            slot = slots.get(descriptor.key)
            if slot is None:
                continue
            cached = self._patch_edge_vertices(descriptor.key)
            edge_rows.append((cached, int(slot)))
            row_count += len(cached)
            for edge, indices in enumerate(self._edge_indices):
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
        destinations = np.empty((0, 2), np.int32)
        sources = np.empty((0, 2), np.int32)
        if edge_rows:
            rows = np.empty((row_count, 5), np.int64)
            offset = 0
            for cached, slot in edge_rows:
                end = offset + len(cached)
                rows[offset:end, :3] = cached[:, :3]
                rows[offset:end, 3] = slot
                rows[offset:end, 4] = cached[:, 3]
                offset = end
            order = np.lexsort(
                (rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0])
            )
            rows = rows[order]
            repeated = np.all(rows[1:, :3] == rows[:-1, :3], axis=1)
            if np.any(repeated):
                row_indices = np.arange(len(rows), dtype=np.int64)
                group_start = np.maximum.accumulate(
                    np.where(
                        np.concatenate(([True], ~repeated)),
                        row_indices,
                        0,
                    )
                )
                destination_rows = rows[1:][repeated]
                source_rows = rows[group_start[1:][repeated]]
                destinations = destination_rows[:, 3:5].astype(np.int32)
                sources = source_rows[:, 3:5].astype(np.int32)
        if (
            len(destinations) > self.max_weld_operations
            or len(stitch) > self.max_stitch_operations
        ):
            raise RuntimeError("Patch edge operation capacity exceeded")
        return (
            destinations,
            sources,
            np.asarray(stitch, np.int32).reshape((-1, 4)),
        )

    @ti.kernel
    def _set_edge_operations(
        self,
        weld_count: ti.i32,
        stitch_count: ti.i32,
        operations: ti.types.ndarray(dtype=ti.i32, ndim=2),
    ):
        self.weld_count[None] = weld_count
        self.stitch_count[None] = stitch_count
        for i in range(weld_count + stitch_count):
            self.edge_operations[i] = ti.Vector(
                [operations[i, 0], operations[i, 1], operations[i, 2], operations[i, 3]]
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
        # Batch all patch uploads into a small fixed number of GPU launches.
        # This matters most while moving, when several new LOD patches can be
        # scheduled in the same frame.
        self.terrain_renderer.upload_patches(frame.uploads)
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
        active = self._active_host
        skirt_masks = self._skirt_masks_host
        stitch_masks = self._stitch_masks_host
        active.fill(0)
        skirt_masks.fill(0)
        stitch_masks.fill(0)
        for descriptor in descriptors:
            slot = slots.get(descriptor.key)
            if slot is not None:
                active[slot] = 1
                skirt_masks[slot] = descriptor.skirt_mask
                stitch_masks[slot] = descriptor.stitch_mask
        edge_signature = tuple(
            sorted(
                (descriptor.key, int(slots[descriptor.key]), descriptor.stitch_mask)
                for descriptor in descriptors
                if slots.get(descriptor.key) is not None
            )
        )
        if edge_signature != self._edge_signature:
            weld_dst, weld_src, stitch = self._build_edge_operations(descriptors, slots)
            weld_count = len(weld_dst)
            stitch_count = len(stitch)
            operations = np.empty((weld_count + stitch_count, 4), np.int32)
            operations[:weld_count, :2] = weld_dst
            operations[:weld_count, 2:] = weld_src
            operations[weld_count:] = stitch
            self._set_edge_operations(
                weld_count,
                stitch_count,
                operations,
            )
            self._edge_signature = edge_signature
        self._render_signature = render_signature
        self.patch_count = int(active.sum())
        active_ids = self._active_ids_host
        active_slots = np.flatnonzero(active)
        active_count = len(active_slots)
        active_ids[:active_count] = active_slots
        self._set_render(
            active,
            skirt_masks,
            stitch_masks,
            active_ids,
            active_count,
        )

    @ti.kernel
    def _set_render(
        self,
        active: ti.types.ndarray(dtype=ti.i32, ndim=1),
        skirt_masks: ti.types.ndarray(dtype=ti.i32, ndim=1),
        stitch_masks: ti.types.ndarray(dtype=ti.i32, ndim=1),
        active_ids: ti.types.ndarray(dtype=ti.i32, ndim=1),
        active_count: ti.i32,
    ):
        for i in range(self.max_patches):
            self.slot_render[i] = active[i]
            self.slot_skirt_mask[i] = skirt_masks[i]
            self.slot_stitch_mask[i] = stitch_masks[i]
        self.active_slot_count[None] = active_count
        for i in range(active_count):
            self.active_slot_ids[i] = active_ids[i]

    @ti.kernel
    def _prepare_frame(
        self,
        anchors: ti.types.ndarray(dtype=ti.f32, ndim=2),
        e: ti.types.vector(3, ti.f32),
        u: ti.types.vector(3, ti.f32),
        n: ti.types.vector(3, ti.f32),
        r: ti.types.vector(3, ti.f32),
        vu: ti.types.vector(3, ti.f32),
        f: ti.types.vector(3, ti.f32),
    ):
        self.clipped_count[None] = 0
        self.tile_overflow[None] = 0
        self.max_tile_candidates[None] = 0
        for i in range(self.active_slot_count[None]):
            slot = self.active_slot_ids[i]
            self.anchor_relative[slot] = ti.Vector(
                [anchors[slot, 0], anchors[slot, 1], anchors[slot, 2]]
            )
        for q in ti.grouped(self.tile_counts):
            self.tile_counts[q] = 0
        # Stale G-buffer data is ignored whenever surface_id is -1, so there
        # is no reason to clear all large attribute buffers every frame.
        # Both the direct path and the same-frame overflow fallback use this
        # atomic depth buffer. Clearing it here avoids a CPU synchronization
        # merely to discover an overflowing tile later in the frame.
        for q in ti.grouped(self.gbuffer_surface_id):
            self.gbuffer_surface_id[q] = -1
            self.gbuffer_surface_cell_id[q] = -1
            self.depth_key[q] = ti.u64(DEPTH_KEY_SENTINEL)
        vertex_count = self.active_slot_count[None] * self.vertices_per_patch
        for active_vertex in range(vertex_count):
            active_index = active_vertex // self.vertices_per_patch
            index = active_vertex % self.vertices_per_patch
            slot = self.active_slot_ids[active_index]
            g = self.anchor_relative[slot] + self.terrain_renderer.offset[slot, index]
            local = ti.Vector([g.dot(e), g.dot(u), g.dot(n)])
            self.view[slot, index] = ti.Vector(
                [local.dot(r), local.dot(vu), local.dot(f)]
            )
            self.frame_normal[slot, index] = self.terrain_renderer.normal[
                slot, index
            ]
            self.frame_material[slot, index] = self.terrain_renderer.material[
                slot, index
            ]
            self.frame_height_m[slot, index] = self.terrain_renderer.height_m[
                slot, index
            ]
            self.frame_cell[slot, index] = self.terrain_renderer.cell[slot, index]

    @ti.kernel
    def _fix_patch_edges(self):
        """Weld current-frame attributes, then stitch mixed-LOD fine edges.

        Position equality alone only provides C0 continuity.  Lighting also
        requires a common normal at every shared vertex.  Each weld group has
        one source and one or more destinations, so the serial accumulation
        below forms one normal from all incident patch estimates before that
        result is copied back to the complete group.
        """

        for operation in range(self.weld_count[None]):
            item = self.edge_operations[operation]
            dst = ti.Vector([item.x, item.y])
            src = ti.Vector([item.z, item.w])
            self.view[dst.x, dst.y] = self.view[src.x, src.y]
            self.frame_height_m[dst.x, dst.y] = self.frame_height_m[src.x, src.y]
            self.frame_material[dst.x, dst.y] = self.frame_material[src.x, src.y]
            self.frame_cell[dst.x, dst.y] = self.frame_cell[src.x, src.y]

        # A cube corner can be shared by more than two patches.  Serialize the
        # small edge-only reduction so all incident normals contribute to the
        # same source without a write race.  The generated patch attributes are
        # restored into frame_normal by _prepare_frame on every frame.
        ti.loop_config(serialize=True)
        for operation in range(self.weld_count[None]):
            item = self.edge_operations[operation]
            dst = ti.Vector([item.x, item.y])
            src = ti.Vector([item.z, item.w])
            self.frame_normal[src.x, src.y] += self.frame_normal[dst.x, dst.y]

        for operation in range(self.weld_count[None]):
            item = self.edge_operations[operation]
            dst = ti.Vector([item.x, item.y])
            src = ti.Vector([item.z, item.w])
            normal_sum = self.frame_normal[src.x, src.y]
            shared_normal = normal_sum / ti.max(normal_sum.norm(), 1.0e-8)
            self.frame_normal[src.x, src.y] = shared_normal
            self.frame_normal[dst.x, dst.y] = shared_normal

        for operation in range(self.stitch_count[None]):
            item = self.edge_operations[self.weld_count[None] + operation]
            slot = item.x
            vertex = item.y
            a = item.z
            b = item.w
            self.view[slot, vertex] = (self.view[slot, a] + self.view[slot, b]) * 0.5
            self.frame_normal[slot, vertex] = (
                self.frame_normal[slot, a]
                + self.frame_normal[slot, b]
            ).normalized()
            self.frame_height_m[slot, vertex] = (
                self.frame_height_m[slot, a]
                + self.frame_height_m[slot, b]
            ) * 0.5
            self.frame_material[slot, vertex] = (
                self.frame_material[slot, a]
                + self.frame_material[slot, b]
            ) * 0.5

    @ti.func
    def _edge_coeff(self, a: ti.template(), b: ti.template()):
        # edge(a,b,p) = A*p.x + B*p.y + C
        return ti.Vector(
            [
                a.y - b.y,
                b.x - a.x,
                (b.y - a.y) * a.x - (b.x - a.x) * a.y,
            ]
        )

    @ti.func
    def _emit(
        self,
        src: ti.i32,
        v: ti.template(),
        n: ti.template(),
        m: ti.template(),
        h: ti.template(),
        c: ti.template(),
        tf: ti.f32,
    ):
        # Every source triangle can produce zero to six clipped triangles.
        # Atomically append each emitted triangle so following passes iterate
        # a dense stream rather than the theoretical capacity.
        s = ti.atomic_add(self.clipped_count[None], 1)
        if s < self.raster_capacity:
            aspect = ti.cast(self.width, ti.f32) / self.height
            sp = ti.Matrix.zero(ti.f32, 3, 3)
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
                sp[k, 0] = projected.x
                sp[k, 1] = projected.y
                sp[k, 2] = p.z
                self.screen[s, k] = ti.Vector([projected.x, projected.y, p.z])

            a = ti.Vector([sp[0, 0], sp[0, 1]])
            b = ti.Vector([sp[1, 0], sp[1, 1]])
            cc = ti.Vector([sp[2, 0], sp[2, 1]])
            area = self._edge(a, b, cc)
            abs_area = ti.abs(area)
            sg = 1.0 if area > 0.0 else -1.0
            self.inv_area[s] = 0.0
            if abs_area > 1e-8:
                self.inv_area[s] = 1.0 / abs_area

            eq0 = self._edge_coeff(b, cc) * sg
            eq1 = self._edge_coeff(cc, a) * sg
            eq2 = self._edge_coeff(a, b) * sg
            self.edge_equation[s, 0] = eq0
            self.edge_equation[s, 1] = eq1
            self.edge_equation[s, 2] = eq2
            self.edge_top_left[s] = ti.Vector(
                [
                    self._top_left(b, cc, sg),
                    self._top_left(cc, a, sg),
                    self._top_left(a, b, sg),
                ]
            )
            self.source[s] = src

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

    @ti.func
    def _clip_code(
        self,
        p: ti.template(),
        near: ti.f32,
        tan_x: ti.f32,
        tan_y: ti.f32,
    ) -> ti.i32:
        code = 0
        if p.z < near:
            code = code | 1
        if p.x < -p.z * tan_x:
            code = code | 2
        if p.x > p.z * tan_x:
            code = code | 4
        if p.y < -p.z * tan_y:
            code = code | 8
        if p.y > p.z * tan_y:
            code = code | 16
        return code

    @ti.kernel
    def _clip(self, near: ti.f32, tf: ti.f32):
        aspect = ti.cast(self.width, ti.f32) / self.height
        tan_x = tf * aspect
        tan_y = tf
        source_count = self.active_slot_count[None] * self.local_triangle_count
        for active_source in range(source_count):
            active_index = active_source // self.local_triangle_count
            slot = self.active_slot_ids[active_index]
            local_id = active_source % self.local_triangle_count
            source_id = slot * self.local_triangle_count + local_id
            skirt_edge = (local_id - self.surface_triangle_count) // (
                self.resolution * 2
            )
            enabled = local_id < self.surface_triangle_count or (
                skirt_edge >= 0
                and (self.slot_skirt_mask[slot] & (1 << skirt_edge)) != 0
            )
            if not enabled:
                continue

            ids = self.local_triangles[local_id]
            p0 = self.view[slot, ids[0]]
            p1 = self.view[slot, ids[1]]
            p2 = self.view[slot, ids[2]]

            # Cull ordinary terrain backfaces before any clipping or attribute
            # interpolation. Skirts remain double-sided as a crack safety net.
            is_skirt = local_id >= self.surface_triangle_count
            front_facing = (p1 - p0).cross(p2 - p0).dot(-p0) < 0.0
            if not is_skirt and not front_facing:
                continue

            c0 = self._clip_code(p0, near, tan_x, tan_y)
            c1 = self._clip_code(p1, near, tan_x, tan_y)
            c2 = self._clip_code(p2, near, tan_x, tan_y)

            # Cohen-Sutherland style trivial reject: all vertices lie outside
            # the same frustum plane.
            if (c0 & c1 & c2) != 0:
                continue

            n0 = self.frame_normal[slot, ids[0]]
            n1 = self.frame_normal[slot, ids[1]]
            n2 = self.frame_normal[slot, ids[2]]
            m0 = self.frame_material[slot, ids[0]]
            m1 = self.frame_material[slot, ids[1]]
            m2 = self.frame_material[slot, ids[2]]
            h0 = self.frame_height_m[slot, ids[0]]
            h1 = self.frame_height_m[slot, ids[1]]
            h2 = self.frame_height_m[slot, ids[2]]
            cell0 = self.frame_cell[slot, ids[0]]
            cell1 = self.frame_cell[slot, ids[1]]
            cell2 = self.frame_cell[slot, ids[2]]

            # The overwhelmingly common case is fully inside the frustum.
            # Emit directly and completely bypass five-plane polygon clipping.
            if (c0 | c1 | c2) == 0:
                self._emit(
                    source_id,
                    ti.Matrix.rows([p0, p1, p2]),
                    ti.Matrix.rows([n0, n1, n2]),
                    ti.Matrix.rows([m0, m1, m2]),
                    ti.Vector([h0, h1, h2]),
                    ti.Vector([cell0, cell1, cell2]),
                    tf,
                )
                continue

            # Only boundary-crossing triangles pay for Sutherland-Hodgman.
            positions = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 3)
            normals = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 3)
            materials = ti.Matrix.zero(ti.f32, MAX_CLIP_VERTICES * 2, 4)
            heights = ti.Vector.zero(ti.f32, MAX_CLIP_VERTICES * 2)
            cells = ti.Vector.zero(ti.i32, MAX_CLIP_VERTICES * 2)
            positions[0, :] = p0
            positions[1, :] = p1
            positions[2, :] = p2
            normals[0, :] = n0
            normals[1, :] = n1
            normals[2, :] = n2
            materials[0, :] = m0
            materials[1, :] = m1
            materials[2, :] = m2
            heights[0] = h0
            heights[1] = h1
            heights[2] = h2
            cells[0] = cell0
            cells[1] = cell1
            cells[2] = cell2

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
                            destination = write_buffer * MAX_CLIP_VERTICES + output_count
                            positions[destination, :] = previous_position + (
                                current_position - previous_position
                            ) * t
                            normals[destination, :] = (
                                normals[previous_index, :]
                                + (
                                    normals[current_index, :]
                                    - normals[previous_index, :]
                                )
                                * t
                            ).normalized()
                            materials[destination, :] = materials[previous_index, :] + (
                                materials[current_index, :]
                                - materials[previous_index, :]
                            ) * t
                            heights[destination] = heights[previous_index] + (
                                heights[current_index] - heights[previous_index]
                            ) * t
                            cells[destination] = (
                                cells[current_index]
                                if current_inside
                                else cells[previous_index]
                            )
                            output_count += 1
                        if current_inside and output_count < MAX_CLIP_VERTICES:
                            destination = write_buffer * MAX_CLIP_VERTICES + output_count
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

    @ti.func
    def _triangle_overlaps_tile(
        self,
        q: ti.i32,
        tile_x: ti.i32,
        tile_y: ti.i32,
    ) -> ti.i32:
        """Conservative triangle/rectangle test using oriented edge planes."""

        x0 = ti.cast(tile_x * TILE_SIZE, ti.f32)
        y0 = ti.cast(tile_y * TILE_SIZE, ti.f32)
        x1 = ti.cast(ti.min((tile_x + 1) * TILE_SIZE, self.width), ti.f32)
        y1 = ti.cast(ti.min((tile_y + 1) * TILE_SIZE, self.height), ti.f32)
        overlaps = 1
        for edge in ti.static(range(3)):
            eq = self.edge_equation[q, edge]
            # Maximum edge value over an axis-aligned rectangle. If even that
            # corner is outside, the whole tile lies outside this triangle edge.
            px = x1 if eq.x >= 0.0 else x0
            py = y1 if eq.y >= 0.0 else y0
            maximum = eq.x * px + eq.y * py + eq.z
            if maximum < 0.0:
                overlaps = 0
        return overlaps

    @ti.kernel
    def _bin(self):
        triangle_count = ti.min(self.clipped_count[None], self.raster_capacity)
        for q in range(triangle_count):
            if self.inv_area[q] <= 0.0:
                continue
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
                    (x0 // TILE_SIZE, x1 // TILE_SIZE + 1),
                    (y0 // TILE_SIZE, y1 // TILE_SIZE + 1),
                ):
                    if self._triangle_overlaps_tile(q, x, y) != 0:
                        k = ti.atomic_add(self.tile_counts[x, y], 1)
                        ti.atomic_max(
                            self.max_tile_candidates[None],
                            k + 1,
                        )
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
    def _surface_color(
        self,
        h: ti.f32,
        material: ti.template(),
        slot: ti.i32,
        mode: ti.i32,
    ):
        fertile = ti.Vector([0.10, 0.34, 0.075])
        arid = ti.Vector([0.44, 0.31, 0.13])
        rock = ti.Vector([0.34, 0.32, 0.30])
        snow = ti.Vector([0.92, 0.95, 0.98])
        sand = ti.Vector([0.72, 0.60, 0.34])
        deep_ocean = ti.Vector([0.012, 0.055, 0.16])
        shallow_ocean = ti.Vector([0.015, 0.34, 0.46])

        color = (
            fertile * material[0]
            + arid * material[1]
            + rock * material[2]
            + snow * material[3]
        )
        if h < 0.0:
            water_depth = ti.min(ti.max(-h / 4200.0, 0.0), 1.0)
            color = shallow_ocean * (1.0 - water_depth) + deep_ocean * water_depth
        elif h < 140.0:
            coast = ti.min(ti.max(h / 140.0, 0.0), 1.0)
            color = sand * (1.0 - coast) + color * coast

        if mode == 1:
            color = self._height_band(h)
        elif mode == 2:
            hue = ti.cast(self.terrain_renderer.slot_level[slot] % 6, ti.f32) / 6.0
            color = ti.Vector(
                [
                    ti.abs(hue * 6.0 - 3.0) - 1.0,
                    2.0 - ti.abs(hue * 6.0 - 2.0),
                    2.0 - ti.abs(hue * 6.0 - 4.0),
                ]
            )
            color = ti.min(ti.max(color, 0.0), 1.0)
        elif mode == 3:
            value = ti.cast((slot * 1103515245 + 12345) & 255, ti.f32) / 255.0
            color = ti.Vector(
                [
                    value,
                    ti.math.fract(value * 0.73 + 0.21),
                    ti.math.fract(value * 0.37 + 0.61),
                ]
            )
        return color

    @ti.func
    def _edge_values(self, q: ti.i32, p: ti.template()):
        e0 = self.edge_equation[q, 0]
        e1 = self.edge_equation[q, 1]
        e2 = self.edge_equation[q, 2]
        return ti.Vector(
            [
                e0.x * p.x + e0.y * p.y + e0.z,
                e1.x * p.x + e1.y * p.y + e1.z,
                e2.x * p.x + e2.y * p.y + e2.z,
            ]
        )

    @ti.func
    def _inside_edges(self, q: ti.i32, e: ti.template()) -> ti.i32:
        top_left = self.edge_top_left[q]
        return ti.cast(
            (e.x > 0.0 or (e.x == 0.0 and top_left.x != 0))
            and (e.y > 0.0 or (e.y == 0.0 and top_left.y != 0))
            and (e.z > 0.0 or (e.z == 0.0 and top_left.z != 0)),
            ti.i32,
        )

    @ti.kernel
    def _raster_pixel(self, mode: ti.i32):
        """Pixel-parallel path for low-overdraw tiles.

        Triangle edge equations and inverse area are precomputed once in
        ``_emit``.  The hot pixel/candidate loop is therefore reduced to three
        fused linear evaluations plus perspective-depth work.
        """

        for x, y in self.gbuffer_surface_id:
            p = ti.Vector([ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5])
            best = 1e30
            src = -1
            cell = -1
            bp = ti.Vector.zero(ti.f32, 3)
            bn = ti.Vector.zero(ti.f32, 3)
            bm = ti.Vector.zero(ti.f32, 4)
            bh = 0.0
            tile_x = x // TILE_SIZE
            tile_y = y // TILE_SIZE
            candidate_count = ti.min(
                self.tile_counts[tile_x, tile_y], MAX_TRIANGLES_PER_TILE
            )
            for k in range(candidate_count):
                q = self.tile_triangles[tile_x, tile_y, k]
                e = self._edge_values(q, p)
                if self._inside_edges(q, e) != 0:
                    a = self.screen[q, 0]
                    b = self.screen[q, 1]
                    c = self.screen[q, 2]
                    l = e * self.inv_area[q]
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
                        best = z
                        src = self.source[q]
                        cell = self.rc[q, vi]
            if src >= 0:
                slot = src // self.local_triangle_count
                self.depth[x, y] = best
                self.gbuffer_surface_id[x, y] = src
                self.gbuffer_surface_cell_id[x, y] = cell
                self.gbuffer_position[x, y] = bp
                self.gbuffer_normal[x, y] = bn
                self.gbuffer_material_weights[x, y] = bm
                self.gbuffer_height_m[x, y] = bh
                self.gbuffer_albedo[x, y] = self._surface_color(
                    bh, bm, slot, mode
                )

    @ti.kernel
    def _raster_depth_direct(self, overflow_only: ti.template()):
        """Rasterize triangle bounds without a fixed per-tile candidate cap.

        The full variant is the high-overdraw path. The overflow-only variant
        repairs tiles truncated by the bounded pixel path in the same frame.
        """

        triangle_count = ti.min(self.clipped_count[None], self.raster_capacity)
        for q in range(triangle_count):
            active = self.inv_area[q] > 0.0
            if ti.static(overflow_only):
                active = active and self.tile_overflow[None] > 0
            if not active:
                continue

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
                for x, y in ti.ndrange((x0, x1 + 1), (y0, y1 + 1)):
                    rasterize = True
                    if ti.static(overflow_only):
                        rasterize = (
                            self.tile_counts[x // TILE_SIZE, y // TILE_SIZE]
                            > MAX_TRIANGLES_PER_TILE
                        )
                    if rasterize:
                        p = ti.Vector(
                            [ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5]
                        )
                        e = self._edge_values(q, p)
                        if self._inside_edges(q, e) != 0:
                            l = e * self.inv_area[q]
                            z = 1.0 / ti.max(
                                l.x / a.z + l.y / b.z + l.z / c.z,
                                1e-20,
                            )
                            depth_bits = ti.bit_cast(z, ti.u32)
                            key = (
                                ti.cast(depth_bits, ti.u64) << DEPTH_KEY_Q_BITS
                            ) | ti.cast(q, ti.u64)
                            ti.atomic_min(self.depth_key[x, y], key)

    @ti.kernel
    def _resolve_raster(self, mode: ti.i32, overflow_only: ti.template()):
        # Resolve the triangle selected by the atomic depth key.  This is one
        # interpolation per covered pixel instead of a second candidate scan.
        for x, y in self.depth:
            key = self.depth_key[x, y]
            resolve = key != ti.u64(DEPTH_KEY_SENTINEL)
            if ti.static(overflow_only):
                resolve = resolve and (
                    self.tile_counts[x // TILE_SIZE, y // TILE_SIZE]
                    > MAX_TRIANGLES_PER_TILE
                )
            if resolve:
                depth_bits = ti.cast(key >> DEPTH_KEY_Q_BITS, ti.u32)
                q = ti.cast(key & ti.u64(DEPTH_KEY_Q_MASK), ti.i32)
                z = ti.bit_cast(depth_bits, ti.f32)
                a, b, c = self.screen[q, 0], self.screen[q, 1], self.screen[q, 2]
                p = ti.Vector([ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5])
                e = self._edge_values(q, p)
                l = e * self.inv_area[q]
                w = ti.Vector(
                    [l.x * z / a.z, l.y * z / b.z, l.z * z / c.z]
                )
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
                src = self.source[q]
                slot = src // self.local_triangle_count
                self.depth[x, y] = z
                self.gbuffer_surface_id[x, y] = src
                self.gbuffer_surface_cell_id[x, y] = self.rc[q, vi]
                self.gbuffer_position[x, y] = bp
                self.gbuffer_normal[x, y] = bn
                self.gbuffer_material_weights[x, y] = bm
                self.gbuffer_height_m[x, y] = bh
                self.gbuffer_albedo[x, y] = self._surface_color(
                    bh, bm, slot, mode
                )

    @ti.kernel
    def _snapshot_terrain_gbuffer(self):
        for q in ti.grouped(self.gbuffer_surface_id):
            self.terrain_position[q] = self.gbuffer_position[q]
            self.terrain_normal[q] = self.gbuffer_normal[q]
            self.terrain_albedo[q] = self.gbuffer_albedo[q]
            self.terrain_surface_id[q] = self.gbuffer_surface_id[q]

    @ti.kernel
    def _shade_surface(
        self,
        sun_global: ti.types.vector(3, ti.f32),
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_transmittance: ti.template(),
        sky_radiance: ti.template(),
        view_right_global: ti.types.vector(3, ti.f32),
        view_up_global: ti.types.vector(3, ti.f32),
        view_forward_global: ti.types.vector(3, ti.f32),
        advanced_ocean_enabled: ti.i32,
        ocean_roughness: ti.f32,
        ocean_f0: ti.f32,
        ocean_sky_strength: ti.f32,
        ocean_absorption: ti.types.vector(3, ti.f32),
        ocean_scattering: ti.types.vector(3, ti.f32),
        ocean_max_depth: ti.f32,
        ocean_refraction_index: ti.f32,
        ocean_refraction_strength: ti.f32,
        ocean_refraction_max_offset_pixels: ti.f32,
        sun_angular_radius: ti.f32,
        tangent_half_fov: ti.f32,
        mode: ti.i32,
    ):
        for q in ti.grouped(self.surface_hdr):
            color = ti.Vector([0.00015, 0.0002, 0.00035])
            if self.gbuffer_surface_id[q] >= 0:
                albedo = self.gbuffer_albedo[q]
                color = albedo
                if mode == 0:
                    normal = self.gbuffer_normal[q].normalized()
                    ndotl = ti.max(normal.dot(sun_global), 0.0)
                    direct_irradiance = (
                        solar_irradiance
                        * sun_transmittance[q]
                        * ndotl
                    )
                    color = albedo * (
                        direct_irradiance / math.pi + sky_radiance[q]
                    )
                    if (
                        self.gbuffer_surface_id[q] == OCEAN_SURFACE_ID
                        and advanced_ocean_enabled != 0
                    ):
                        position = self.gbuffer_position[q]
                        view_direction = -(
                            view_right_global * position.x
                            + view_up_global * position.y
                            + view_forward_global * position.z
                        ).normalized()
                        ndotv = ti.max(normal.dot(view_direction), 1.0e-4)
                        half_sum = view_direction + sun_global
                        half_vector = half_sum / ti.max(half_sum.norm(), 1.0e-5)
                        ndoth = ti.max(normal.dot(half_vector), 0.0)
                        vdoth = ti.max(view_direction.dot(half_vector), 0.0)

                        one_minus_v = 1.0 - ndotv
                        fresnel_view = ocean_f0 + (1.0 - ocean_f0) * (
                            one_minus_v * one_minus_v * one_minus_v
                            * one_minus_v * one_minus_v
                        )
                        one_minus_h = 1.0 - vdoth
                        fresnel_sun = ocean_f0 + (1.0 - ocean_f0) * (
                            one_minus_h * one_minus_h * one_minus_h
                            * one_minus_h * one_minus_h
                        )

                        base_alpha = ocean_roughness * ocean_roughness
                        solar_slope_variance = ti.tan(sun_angular_radius) ** 2
                        alpha2 = (
                            base_alpha * base_alpha
                            + self.gbuffer_ocean_slope_variance[q]
                            + solar_slope_variance
                        )
                        alpha2 = ti.math.clamp(alpha2, 1.0e-5, 1.0)
                        denominator = ndoth * ndoth * (alpha2 - 1.0) + 1.0
                        distribution = alpha2 / ti.max(
                            math.pi * denominator * denominator,
                            1.0e-5,
                        )
                        effective_roughness = ti.sqrt(ti.sqrt(alpha2))
                        geometry_k = (effective_roughness + 1.0) ** 2 / 8.0
                        geometry_v = ndotv / ti.max(
                            ndotv * (1.0 - geometry_k) + geometry_k,
                            1.0e-5,
                        )
                        geometry_l = ndotl / ti.max(
                            ndotl * (1.0 - geometry_k) + geometry_k,
                            1.0e-5,
                        )
                        sun_specular = (
                            solar_irradiance
                            * sun_transmittance[q]
                            * distribution
                            * geometry_v
                            * geometry_l
                            * fresnel_sun
                            * ndotl
                            / ti.max(4.0 * ndotv * ndotl, 1.0e-4)
                        )
                        reflected_sky = (
                            sky_radiance[q]
                            * fresnel_view
                            * ocean_sky_strength
                        )
                        water_depth = ti.min(
                            self.gbuffer_water_depth_m[q],
                            ocean_max_depth,
                        )
                        seabed_albedo = self.gbuffer_seabed_albedo[q]
                        seabed_normal = self.gbuffer_seabed_normal[q].normalized()

                        # Refract the camera ray from air into water, project a
                        # bounded endpoint back to the opaque terrain snapshot,
                        # and bilinearly reconstruct the seabed.  Missing or
                        # disoccluded samples fall back to the same-pixel
                        # seabed captured by OceanRenderer.
                        incident_view = position.normalized()
                        normal_view = ti.Vector(
                            [
                                normal.dot(view_right_global),
                                normal.dot(view_up_global),
                                normal.dot(view_forward_global),
                            ]
                        ).normalized()
                        cos_incident = ti.max(
                            -incident_view.dot(normal_view),
                            0.0,
                        )
                        eta = 1.0 / ocean_refraction_index
                        refract_discriminant = 1.0 - eta * eta * (
                            1.0 - cos_incident * cos_incident
                        )
                        refracted_view = incident_view
                        if refract_discriminant > 0.0:
                            refracted_view = (
                                eta * incident_view
                                + (
                                    eta * cos_incident
                                    - ti.sqrt(refract_discriminant)
                                )
                                * normal_view
                            ).normalized()
                        refracted_view = (
                            incident_view * (1.0 - ocean_refraction_strength)
                            + refracted_view * ocean_refraction_strength
                        ).normalized()
                        vertical_depth = water_depth * ti.max(
                            -incident_view.dot(normal_view),
                            0.05,
                        )
                        refracted_distance = ti.min(
                            vertical_depth
                            / ti.max(-refracted_view.dot(normal_view), 0.05),
                            ocean_max_depth * 2.0,
                        )
                        endpoint = position + refracted_view * refracted_distance
                        if endpoint.z > 0.1 and ocean_refraction_strength > 0.0:
                            aspect = ti.cast(self.width, ti.f32) / self.height
                            sample_position = ti.Vector(
                                [
                                    (
                                        endpoint.x
                                        / (endpoint.z * tangent_half_fov * aspect)
                                        * 0.5
                                        + 0.5
                                    )
                                    * self.width
                                    - 0.5,
                                    (
                                        endpoint.y
                                        / (endpoint.z * tangent_half_fov)
                                        * 0.5
                                        + 0.5
                                    )
                                    * self.height
                                    - 0.5,
                                ]
                            )
                            pixel_delta = sample_position - ti.cast(q, ti.f32)
                            delta_length = pixel_delta.norm()
                            if delta_length > ocean_refraction_max_offset_pixels:
                                pixel_delta *= ocean_refraction_max_offset_pixels / ti.max(
                                    delta_length,
                                    1.0e-5,
                                )
                                sample_position = ti.cast(q, ti.f32) + pixel_delta
                            sample_base = ti.cast(ti.floor(sample_position), ti.i32)
                            sample_fraction = sample_position - ti.cast(
                                sample_base,
                                ti.f32,
                            )
                            accumulated_albedo = ti.Vector.zero(ti.f32, 3)
                            accumulated_normal = ti.Vector.zero(ti.f32, 3)
                            accumulated_position = ti.Vector.zero(ti.f32, 3)
                            accumulated_weight = 0.0
                            for oy in ti.static(range(2)):
                                for ox in ti.static(range(2)):
                                    sample = sample_base + ti.Vector([ox, oy])
                                    inside = (
                                        sample.x >= 0
                                        and sample.x < self.width
                                        and sample.y >= 0
                                        and sample.y < self.height
                                    )
                                    safe_sample = ti.Vector(
                                        [
                                            ti.math.clamp(sample.x, 0, self.width - 1),
                                            ti.math.clamp(sample.y, 0, self.height - 1),
                                        ]
                                    )
                                    if inside and self.terrain_surface_id[safe_sample] >= 0:
                                        candidate_position = self.terrain_position[
                                            safe_sample
                                        ]
                                        candidate_distance = (
                                            candidate_position - position
                                        ).norm()
                                        valid_depth = (
                                            candidate_position.z > position.z + 0.01
                                            and candidate_distance
                                            <= ocean_max_depth * 2.0
                                        )
                                        if valid_depth:
                                            wx = ti.select(
                                                ox == 0,
                                                1.0 - sample_fraction.x,
                                                sample_fraction.x,
                                            )
                                            wy = ti.select(
                                                oy == 0,
                                                1.0 - sample_fraction.y,
                                                sample_fraction.y,
                                            )
                                            weight = wx * wy
                                            accumulated_albedo += (
                                                self.terrain_albedo[safe_sample]
                                                * weight
                                            )
                                            accumulated_normal += (
                                                self.terrain_normal[safe_sample]
                                                * weight
                                            )
                                            accumulated_position += (
                                                candidate_position * weight
                                            )
                                            accumulated_weight += weight
                            if accumulated_weight > 0.25:
                                inverse_weight = 1.0 / accumulated_weight
                                seabed_albedo = accumulated_albedo * inverse_weight
                                seabed_normal = (
                                    accumulated_normal * inverse_weight
                                ).normalized()
                                refracted_seabed_position = (
                                    accumulated_position * inverse_weight
                                )
                                water_depth = ti.min(
                                    (refracted_seabed_position - position).norm(),
                                    ocean_max_depth,
                                )
                        water_extinction = ocean_absorption + ocean_scattering
                        water_transmission = ti.exp(
                            -water_extinction * water_depth
                        )
                        seabed_ndotl = ti.max(
                            seabed_normal.dot(sun_global),
                            0.0,
                        )
                        seabed_direct = (
                            solar_irradiance
                            * sun_transmittance[q]
                            * seabed_ndotl
                        )
                        lit_seabed = seabed_albedo * (
                            seabed_direct / math.pi + sky_radiance[q]
                        )
                        seabed = lit_seabed * water_transmission

                        # The old path added a fixed blue value here, making
                        # water self-emissive on the planet's night side.
                        # Scattering now consumes only shared incident light.
                        scattering_albedo = ocean_scattering / ti.max(
                            water_extinction,
                            1.0e-6,
                        )
                        incident_water_radiance = (
                            sky_radiance[q]
                            + direct_irradiance / (4.0 * math.pi)
                        )
                        water_scattering = (
                            incident_water_radiance
                            * scattering_albedo
                            * (1.0 - water_transmission)
                        )
                        water_body = (seabed + water_scattering) * (
                            1.0 - fresnel_view
                        )
                        color = water_body + reflected_sky + sun_specular
            self.surface_hdr[q] = ti.max(color, 0.0)

    def render(
        self,
        planet: PlanetModel,
        camera: PlanetCamera,
        lighting: LightingState,
        surface_albedo: tuple[float, float, float],
        exposure_ev: float,
        time_seconds: float = 0.0,
    ) -> None:
        self.planet_radius = planet.radius_m
        anchors = self._anchors_host
        for descriptor in self._render_descriptors:
            slot = self._render_slots.get(descriptor.key)
            if slot is not None:
                anchors[slot] = (
                    descriptor.anchor_global - camera.position_global
                ).astype(np.float32)
        frame = camera.frame(planet)
        r, vu, f = camera.view_basis_local()
        tf = math.tan(math.radians(camera.vertical_fov_degrees) / 2)
        near = 0.1
        sun_local = frame.global_to_local_direction(lighting.sun_direction_global)
        # Keep one raster strategy for the complete frame. A probe may update
        # ``_use_triangle_raster`` below, but that choice takes effect on the
        # next frame so clear/bin/raster always agree on their data layout.
        triangle_raster_this_frame = self._use_triangle_raster
        self._prepare_frame(
            anchors,
            tuple(frame.east),
            tuple(frame.up),
            tuple(frame.north),
            tuple(r),
            tuple(vu),
            tuple(f),
        )
        self._fix_patch_edges()
        self._clip(near, tf)
        self._raster_probe_frame += 1
        probe_this_frame = self._raster_probe_frame % RASTER_PROBE_INTERVAL == 0
        # Pixel rasterization needs bins every frame. The direct path builds
        # them only while probing whether tile density has fallen again.
        if not triangle_raster_this_frame or probe_this_frame:
            self._bin()
        if probe_this_frame:
            # Reading one scalar every few frames avoids a per-frame
            # synchronization point while still switching to the
            # triangle-driven path as soon as tile overdraw becomes large.
            ti.sync()
            self.last_max_tile_candidates = int(self.max_tile_candidates[None])
            self.last_tile_overflow = int(self.tile_overflow[None])
            self._use_triangle_raster = (
                self.last_max_tile_candidates >= TRIANGLE_RASTER_THRESHOLD
            )
        if triangle_raster_this_frame:
            self._raster_depth_direct(False)
            self._resolve_raster(self.debug_view, False)
        else:
            self._raster_pixel(self.debug_view)
            # The bounded fast path may overflow in dense horizon tiles.
            # Recompute only those pixels from the complete triangle stream,
            # then overwrite their incomplete G-buffer values before shading.
            self._raster_depth_direct(True)
            self._resolve_raster(self.debug_view, True)
        self._snapshot_terrain_gbuffer()
        camera_radius = float(np.linalg.norm(camera.position_global))
        camera_altitude = camera_radius - planet.radius_m
        self.ocean_renderer.rasterize(
            self.depth,
            self.gbuffer_position,
            self.gbuffer_normal,
            self.gbuffer_albedo,
            self.gbuffer_material_weights,
            self.gbuffer_height_m,
            self.gbuffer_water_depth_m,
            self.gbuffer_seabed_albedo,
            self.gbuffer_seabed_normal,
            self.gbuffer_ocean_slope_variance,
            self.gbuffer_surface_id,
            self.gbuffer_surface_cell_id,
            planet.radius_m,
            camera_altitude,
            (frame.east, frame.up, frame.north),
            (r, vu, f),
            tf,
            time_seconds,
            self.debug_view,
        )
        self.atmosphere_renderer.update(
            planet.radius_m,
            camera_radius,
            sun_local,
            lighting.solar_irradiance,
            lighting.sun_angular_radius_degrees,
        )
        self.atmosphere_renderer.prepare_frame(
            self.gbuffer_position,
            self.gbuffer_surface_id,
            planet.radius_m,
            camera_radius,
            sun_local,
            lighting.solar_irradiance,
            lighting.sun_angular_radius_degrees,
            (r, vu, f),
            tf,
        )
        self._shade_surface(
            tuple(lighting.sun_direction_global),
            lighting.solar_irradiance,
            self.atmosphere_renderer.surface_sun_transmittance,
            self.atmosphere_renderer.surface_sky_radiance,
            tuple(frame.local_to_global_direction(r)),
            tuple(frame.local_to_global_direction(vu)),
            tuple(frame.local_to_global_direction(f)),
            int(self.ocean_renderer.config.enabled),
            self.ocean_renderer.config.roughness,
            self.ocean_renderer.config.dielectric_f0,
            self.ocean_renderer.config.sky_reflection_strength,
            self.ocean_renderer.config.absorption_per_m,
            self.ocean_renderer.config.scattering_per_m,
            self.ocean_renderer.config.max_visible_depth_m,
            self.ocean_renderer.config.refraction_index,
            self.ocean_renderer.config.refraction_strength,
            self.ocean_renderer.config.refraction_max_offset_pixels,
            math.radians(lighting.sun_angular_radius_degrees),
            tf,
            self.debug_view,
        )
        global_view_basis = (
            frame.local_to_global_direction(r),
            frame.local_to_global_direction(vu),
            frame.local_to_global_direction(f),
        )
        self.space_renderer.render(global_view_basis, tf)
        self.atmosphere_renderer.composite(
            self.surface_hdr,
            self.space_renderer.hdr,
            self.gbuffer_surface_id,
            self.gbuffer_position,
            self.hdr,
            planet.radius_m,
            camera_radius,
            sun_local,
            lighting.solar_irradiance,
            lighting.sun_disk_radiance,
            lighting.sun_angular_radius_degrees,
            (r, vu, f),
            tf,
            self.space_renderer.config.contrast_start,
            self.space_renderer.config.contrast_end,
            self.atmosphere_diagnostic_view,
        )
        diagnostic = self.atmosphere_diagnostic_view
        self.postprocessor.process(
            self.hdr,
            self.display,
            exposure_ev,
            bloom=diagnostic.uses_bloom,
            tone_map=diagnostic.is_radiance,
        )

    def warmup_raster_paths(
        self,
        planet: PlanetModel,
        camera: PlanetCamera,
        lighting: LightingState,
        surface_albedo: tuple[float, float, float],
        exposure_ev: float,
    ) -> None:
        """Compile both raster strategies before an interactive frame can select one."""

        previous = self._use_triangle_raster
        for triangle_raster in (False, True):
            self._use_triangle_raster = triangle_raster
            self.render(planet, camera, lighting, surface_albedo, exposure_ev)
        ti.sync()
        self._use_triangle_raster = previous
        self._raster_probe_frame = 0

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
            self.last_tile_overflow,
        )

    def display_numpy(self) -> np.ndarray:
        return self.display.to_numpy()

    def hdr_numpy(self) -> np.ndarray:
        return self.hdr.to_numpy()
