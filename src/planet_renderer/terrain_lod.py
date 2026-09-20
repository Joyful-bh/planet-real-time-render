"""持久 Mixed-LOD 选择器：SSE、滞回与相邻层级平衡。"""

from __future__ import annotations

import math
import time

import numpy as np

from .camera import PlanetCamera
from .height import TerrainHeightModel
from .planet import PlanetModel, Vec3d, normalize
from .terrain_types import PatchKey


def cube_face_direction(face: int, u: float, v: float) -> Vec3d:
    return normalize(cube_face_vector(face, u, v))


def cube_face_vector(face: int, u: float, v: float) -> Vec3d:
    mappings = (
        (1.0, v, -u),
        (-1.0, v, u),
        (u, 1.0, -v),
        (u, -1.0, v),
        (u, v, 1.0),
        (-u, v, -1.0),
    )
    return np.asarray(mappings[face], np.float64)


def direction_to_cube_face_uv(direction: Vec3d) -> tuple[int, float, float]:
    x, y, z = direction
    ax, ay, az = abs(x), abs(y), abs(z)
    if max(ax, ay, az) <= 1e-30:
        raise ValueError("direction must be non-zero")
    if ax >= ay and ax >= az:
        return (0, -z / ax, y / ax) if x >= 0 else (1, z / ax, y / ax)
    if ay >= ax and ay >= az:
        return (2, x / ay, -z / ay) if y >= 0 else (3, x / ay, z / ay)
    return (4, x / az, y / az) if z >= 0 else (5, -x / az, y / az)


def patch_uv_bounds(key: PatchKey) -> tuple[float, float, float, float]:
    size = 2.0 / (1 << key.level)
    return (
        -1 + key.x * size,
        -1 + (key.x + 1) * size,
        -1 + key.y * size,
        -1 + (key.y + 1) * size,
    )


def patch_center_direction(key: PatchKey) -> Vec3d:
    u0, u1, v0, v1 = patch_uv_bounds(key)
    return cube_face_direction(key.face, (u0 + u1) * 0.5, (v0 + v1) * 0.5)


class MixedLodSelector:
    def __init__(
        self,
        planet: PlanetModel,
        height_model: TerrainHeightModel,
        resolution: int,
        max_level: int = 16,
        split_sse: float = 64.0,
        merge_sse: float = 32.0,
        max_leaves: int = 180,
        max_changes: int = 8,
    ):
        if merge_sse >= split_sse:
            raise ValueError("merge_sse 必须小于 split_sse")
        self.planet = planet
        self.height_model = height_model
        self.resolution = resolution
        self.max_level = max_level
        self.split_sse = split_sse
        self.merge_sse = merge_sse
        self.max_leaves = max_leaves
        self.max_changes = max_changes
        self.metadata_cache_capacity = max(1024, max_leaves * 8)
        self.leaves = frozenset(PatchKey(face, 0, 0, 0) for face in range(6))
        self.last_ms = 0.0
        self.last_changes = max_changes
        self._center_cache: dict[PatchKey, Vec3d] = {}
        self._terrain_error_cache: dict[PatchKey, float] = {}
        self._terrain_height_cache: dict[PatchKey, float] = {}
        self._angular_radius_cache: dict[PatchKey, float] = {}
        self._sse_frame_signature: (
            tuple[float, float, float, float, int] | None
        ) = None
        self._sse_frame_cache: dict[PatchKey, float] = {}

    def center(self, key: PatchKey) -> Vec3d:
        value = self._center_cache.get(key)
        if value is None:
            value = patch_center_direction(key)
            self._center_cache[key] = value
        return value

    def angular_radius(self, key: PatchKey) -> float:
        value = self._angular_radius_cache.get(key)
        if value is None:
            u0, u1, v0, v1 = patch_uv_bounds(key)
            center = self.center(key)
            value = max(
                math.acos(
                    float(
                        np.clip(
                            np.dot(center, cube_face_direction(key.face, u, v)),
                            -1.0,
                            1.0,
                        )
                    )
                )
                for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
            )
            self._angular_radius_cache[key] = value
        return value

    def sse(self, key: PatchKey, camera: PlanetCamera, viewport_height: int) -> float:
        signature = (
            float(camera.position_global[0]),
            float(camera.position_global[1]),
            float(camera.position_global[2]),
            float(camera.vertical_fov_degrees),
            int(viewport_height),
        )
        if signature != self._sse_frame_signature:
            self._sse_frame_signature = signature
            self._sse_frame_cache.clear()
        cached_sse = self._sse_frame_cache.get(key)
        if cached_sse is not None:
            return cached_sse

        center_direction = self.center(key)
        center = center_direction * self.planet.radius_m
        distance = max(float(np.linalg.norm(center - camera.position_global)), 1.0)
        focal = viewport_height / (
            2 * math.tan(math.radians(camera.vertical_fov_degrees) * 0.5)
        )
        sphere_error = (
            self.planet.radius_m * 2.4 / ((1 << key.level) * self.resolution)
        )
        terrain_error = self._terrain_error_cache.get(key)
        if terrain_error is None:
            terrain_error = self.height_model.estimate_error_m(
                center_direction,
                key.level,
                self.resolution,
                self.planet.radius_m,
            )
            self._terrain_error_cache[key] = terrain_error
        terrain_height = self._terrain_height_cache.get(key)
        if terrain_height is None:
            terrain_height = self.height_model.sample_height_m(center_direction)
            self._terrain_height_cache[key] = terrain_height

        # Patches fully behind the spherical horizon do not need refinement.
        # They remain covered by resident ancestors, so turning the camera
        # cannot expose a hole and frustum visibility remains independent from
        # residency. Local height plus unresolved relief conservatively extends
        # the horizon for elevated terrain on the deliberately small test planet.
        camera_radius = float(np.linalg.norm(camera.position_global))
        if camera_radius > self.planet.radius_m + 1.0:
            camera_direction = camera.position_global / camera_radius
            horizon = math.acos(
                float(np.clip(self.planet.radius_m / camera_radius, -1.0, 1.0))
            )
            local_elevation = max(terrain_height + terrain_error, 0.0)
            relief = math.acos(
                self.planet.radius_m
                / (self.planet.radius_m + local_elevation)
            )
            separation = math.acos(
                float(np.clip(np.dot(center_direction, camera_direction), -1.0, 1.0))
            )
            if separation > horizon + self.angular_radius(key) + relief:
                self._sse_frame_cache[key] = 0.0
                return 0.0
        geometric_error = math.hypot(sphere_error, terrain_error)
        result = geometric_error * focal / distance
        self._sse_frame_cache[key] = result
        return result

    def select(self, camera: PlanetCamera, viewport_height: int) -> frozenset[PatchKey]:
        started = time.perf_counter()
        previous = self.leaves
        leaves = set(previous)
        changes = 0
        parents = {key.parent() for key in leaves if key.level > 0}
        merge = []
        for parent in parents:
            if parent is not None and all(
                child in leaves for child in parent.children()
            ):
                error = self.sse(parent, camera, viewport_height)
                # Do not merge away a transition-ring patch required by the
                # 2:1 invariant. Otherwise the merge pass and balancing pass
                # undo one another within the same update and falsely report
                # a stable tree with large unresolved SSE.
                preserves_balance = all(
                    (neighbor := self._neighbor(leaves, parent, edge)) is None
                    or neighbor.level <= parent.level + 1
                    for edge in range(4)
                )
                if error < self.merge_sse and preserves_balance:
                    merge.append((error, parent))
        for _, parent in sorted(merge):
            if changes >= self.max_changes:
                break
            if not all(child in leaves for child in parent.children()):
                continue
            if self._merge_preserves_balance(leaves, parent):
                leaves.difference_update(parent.children())
                leaves.add(parent)
                changes += 1
        split = sorted(
            (
                (self.sse(key, camera, viewport_height), key)
                for key in leaves
                if key.level < self.max_level
            ),
            reverse=True,
        )
        for error, key in split:
            if (
                changes >= self.max_changes
                or error <= self.split_sse
                or len(leaves) + 3 > self.max_leaves
            ):
                break
            if key in leaves:
                balanced, split_count = self._try_balanced_split(
                    leaves,
                    key,
                    self.max_changes - changes,
                )
                if split_count:
                    leaves = balanced
                    changes += split_count
        self.leaves = frozenset(leaves)
        self._trim_metadata_cache()
        self.last_changes = 0 if self.leaves == previous else max(changes, 1)
        self.last_ms = (time.perf_counter() - started) * 1000
        return self.leaves

    def _trim_metadata_cache(self) -> None:
        """Bound CPU-only patch metadata during long-distance flight."""

        excess = len(self._center_cache) - self.metadata_cache_capacity
        if excess <= 0:
            return
        protected = self.leaves
        victims = (
            key for key in tuple(self._center_cache) if key not in protected
        )
        for key in victims:
            self._center_cache.pop(key, None)
            self._terrain_error_cache.pop(key, None)
            self._terrain_height_cache.pop(key, None)
            self._angular_radius_cache.pop(key, None)
            excess -= 1
            if excess <= 0:
                break

    def _leaf_at(self, leaves: set[PatchKey], direction: Vec3d) -> PatchKey | None:
        face, u, v = direction_to_cube_face_uv(direction)
        for level in range(self.max_level + 1):
            side = 1 << level
            x = min(max(int((u * 0.5 + 0.5) * side), 0), side - 1)
            y = min(max(int((v * 0.5 + 0.5) * side), 0), side - 1)
            key = PatchKey(face, level, x, y)
            if key in leaves:
                return key
        return None

    @staticmethod
    def _leaf_code(face: int, level: int, x: int, y: int) -> int:
        return (face << 47) | (level << 42) | (x << 21) | y

    def _build_leaf_index(self, leaves: set[PatchKey]) -> dict[int, PatchKey]:
        """Build an integer-key lookup for repeated adjacency queries."""

        return {
            self._leaf_code(key.face, key.level, key.x, key.y): key
            for key in leaves
        }

    def _leaf_at_face_uv_indexed(
        self,
        index: dict[int, PatchKey],
        face: int,
        u: float,
        v: float,
    ) -> PatchKey | None:
        scale = 1 << self.max_level
        gx = min(max(int((u * 0.5 + 0.5) * scale), 0), scale - 1)
        gy = min(max(int((v * 0.5 + 0.5) * scale), 0), scale - 1)
        for level in range(self.max_level + 1):
            shift = self.max_level - level
            key = index.get(
                self._leaf_code(face, level, gx >> shift, gy >> shift)
            )
            if key is not None:
                return key
        return None

    def _neighbor_indexed(
        self,
        index: dict[int, PatchKey],
        key: PatchKey,
        edge: int,
    ) -> PatchKey | None:
        u0, u1, v0, v1 = patch_uv_bounds(key)
        eps = 2.0e-8
        u, v = (
            ((u0 + u1) * 0.5, v0 - eps),
            (u1 + eps, (v0 + v1) * 0.5),
            ((u0 + u1) * 0.5, v1 + eps),
            (u0 - eps, (v0 + v1) * 0.5),
        )[edge]
        if -1.0 <= u <= 1.0 and -1.0 <= v <= 1.0:
            return self._leaf_at_face_uv_indexed(index, key.face, u, v)
        face, mapped_u, mapped_v = direction_to_cube_face_uv(
            cube_face_vector(key.face, u, v)
        )
        return self._leaf_at_face_uv_indexed(
            index,
            face,
            mapped_u,
            mapped_v,
        )

    def _is_balanced(self, leaves: set[PatchKey]) -> bool:
        index = self._build_leaf_index(leaves)
        for key in leaves:
            for edge in range(4):
                neighbor = self._neighbor_indexed(index, key, edge)
                if neighbor is not None and abs(key.level - neighbor.level) > 1:
                    return False
        return True

    def _merge_preserves_balance(
        self,
        leaves: set[PatchKey],
        parent: PatchKey,
    ) -> bool:
        """Check the complete outer boundary of a proposed parent merge."""

        index = self._build_leaf_index(leaves)
        children = parent.children()
        outer_edges = ((0, 3), (0, 1), (2, 3), (2, 1))
        for child, edges in zip(children, outer_edges):
            for edge in edges:
                neighbor = self._neighbor_indexed(index, child, edge)
                if neighbor is not None and neighbor.level > parent.level + 1:
                    return False
        return True

    def _neighbor(
        self, leaves: set[PatchKey], key: PatchKey, edge: int
    ) -> PatchKey | None:
        u0, u1, v0, v1 = patch_uv_bounds(key)
        eps = 2e-8
        uv = (
            ((u0 + u1) * 0.5, v0 - eps),
            (u1 + eps, (v0 + v1) * 0.5),
            ((u0 + u1) * 0.5, v1 + eps),
            (u0 - eps, (v0 + v1) * 0.5),
        )[edge]
        return self._leaf_at(leaves, cube_face_vector(key.face, *uv))

    def _try_balanced_split(
        self,
        leaves: set[PatchKey],
        key: PatchKey,
        operation_budget: int,
    ) -> tuple[set[PatchKey], int]:
        """Split one leaf plus the minimal coarse-neighbor transition ring.

        The input tree is already 2:1 balanced. Before splitting a leaf, only
        its immediate coarser neighbors can become invalid. Refining those
        recursively is both bounded by ``operation_budget`` and much cheaper
        than rescanning the complete leaf set after every split.
        """

        trial = set(leaves)
        operations = 0
        visiting: set[PatchKey] = set()

        def split_with_neighbors(target: PatchKey) -> bool:
            nonlocal operations
            if target not in trial:
                return True
            if target in visiting or target.level >= self.max_level:
                return False
            visiting.add(target)
            for edge in range(4):
                neighbor = self._neighbor(trial, target, edge)
                if neighbor is not None and neighbor.level < target.level:
                    if not split_with_neighbors(neighbor):
                        visiting.remove(target)
                        return False
            if operations >= operation_budget or len(trial) + 3 > self.max_leaves:
                visiting.remove(target)
                return False
            trial.remove(target)
            trial.update(target.children())
            operations += 1
            visiting.remove(target)
            return True

        if operation_budget <= 0 or not split_with_neighbors(key):
            return leaves, 0
        return trial, operations

    def stitch_mask(
        self,
        key: PatchKey,
        render_keys: set[PatchKey],
    ) -> int:
        return self.boundary_masks(key, render_keys)[1]

    def skirt_mask(
        self,
        key: PatchKey,
        render_keys: set[PatchKey],
    ) -> int:
        return self.boundary_masks(key, render_keys)[0]

    def boundary_masks(
        self, key: PatchKey, render_keys: set[PatchKey]
    ) -> tuple[int, int]:
        index = self._build_leaf_index(render_keys)
        return self._boundary_masks_indexed(key, index)

    def _boundary_masks_indexed(
        self,
        key: PatchKey,
        index: dict[int, PatchKey],
    ) -> tuple[int, int]:
        skirt = 0
        stitch = 0
        for edge in range(4):
            neighbor = self._neighbor_indexed(index, key, edge)
            if neighbor is None:
                skirt |= 1 << edge
            elif neighbor.level == key.level - 1:
                stitch |= 1 << edge
        return skirt, stitch

    def boundary_masks_all(
        self,
        render_keys: set[PatchKey],
    ) -> dict[PatchKey, tuple[int, int]]:
        """Compute all boundary masks with one compact leaf index."""

        index = self._build_leaf_index(render_keys)
        return {
            key: self._boundary_masks_indexed(key, index)
            for key in render_keys
        }
