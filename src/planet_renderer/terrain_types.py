"""地形子系统共享的轻量数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


@dataclass(frozen=True, order=True)
class PatchKey:
    face: int
    level: int
    x: int
    y: int

    def children(self) -> tuple["PatchKey", ...]:
        x, y, level = self.x * 2, self.y * 2, self.level + 1
        return tuple(
            PatchKey(self.face, level, x + dx, y + dy)
            for dy in range(2)
            for dx in range(2)
        )

    def parent(self) -> "PatchKey | None":
        if self.level == 0:
            return None
        return PatchKey(self.face, self.level - 1, self.x // 2, self.y // 2)

    def is_ancestor_of(self, other: "PatchKey") -> bool:
        if self.face != other.face or self.level > other.level:
            return False
        shift = other.level - self.level
        return other.x >> shift == self.x and other.y >> shift == self.y


class PatchState(IntEnum):
    UNLOADED = 0
    REQUESTED = 1
    READY = 2
    GPU_RESIDENT = 3


@dataclass(frozen=True)
class TerrainPatchRenderDescriptor:
    """Immutable description of one patch needed by a renderer.

    The descriptor deliberately contains no GPU handle. Slot assignment is a
    streaming/resource-management concern and is carried separately by
    :class:`TerrainFrame` so terrain selection can run without importing or
    calling a renderer.
    """

    key: PatchKey
    anchor_global: np.ndarray
    sse: float
    priority: float
    skirt_mask: int = 0
    stitch_mask: int = 0

    @property
    def face(self) -> int:
        return self.key.face

    @property
    def level(self) -> int:
        return self.key.level

    @property
    def x(self) -> int:
        return self.key.x

    @property
    def y(self) -> int:
        return self.key.y


@dataclass(frozen=True)
class PatchUploadRequest:
    """A patch that should be uploaded into one renderer slot."""

    slot: int
    descriptor: TerrainPatchRenderDescriptor


@dataclass(frozen=True)
class PatchReleaseRequest:
    """A renderer slot that can be released before new uploads."""

    slot: int


@dataclass(frozen=True)
class TerrainDebugStats:
    desired_patches: int = 0
    resident_patches: int = 0
    render_patches: int = 0
    requested_patches: int = 0
    ready_patches: int = 0
    min_lod: int = 0
    max_lod: int = 0
    selection_ms: float = 0.0
    build_ms: float = 0.0
    upload_ms: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass(frozen=True)
class TerrainFrame:
    """Terrain state and renderer operations produced for one frame."""

    desired: frozenset[PatchKey]
    resident: frozenset[PatchKey]
    render: tuple[TerrainPatchRenderDescriptor, ...]
    stats: TerrainDebugStats
    uploads: tuple[PatchUploadRequest, ...] = ()
    releases: tuple[PatchReleaseRequest, ...] = ()
    render_slots: tuple[tuple[PatchKey, int], ...] = ()
