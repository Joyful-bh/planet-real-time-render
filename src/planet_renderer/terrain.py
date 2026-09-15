"""Terrain facade combining LOD selection, streaming and surface queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .camera import PlanetCamera
from .height import DemHeightProvider, HeightProvider, ProceduralHeightProvider
from .planet import PlanetModel, Vec3d
from .surface import SurfaceDescriptor, describe_surface, surface_cell_id
from .terrain_lod import (MixedLodSelector, cube_face_direction,
                          direction_to_cube_face_uv)
from .terrain_streaming import TerrainTileManager
from .terrain_types import (PatchKey, PatchReleaseRequest, PatchState,
                            PatchUploadRequest, TerrainDebugStats,
                            TerrainFrame, TerrainPatchRenderDescriptor)


@dataclass(frozen=True)
class PatchResidencyEvent:
    added: tuple[PatchKey, ...]
    removed: tuple[PatchKey, ...]


class SurfaceCoverageConsumer(Protocol):
    def on_patch_residency_changed(self, event: PatchResidencyEvent) -> None: ...


@dataclass(frozen=True)
class TerrainSettings:
    patch_resolution: int = 12
    max_level: int = 16
    split_sse_pixels: float = 64.0
    merge_sse_pixels: float = 32.0
    max_desired_patches: int = 180
    max_gpu_patches: int = 256
    build_budget_per_frame: int = 8
    upload_budget_per_frame: int = 4
    lod_changes_per_update: int = 8
    cache_capacity: int = 1024
    selection_interval_s: float = 0.1
    canonical_surface_level: int = 14

    def __post_init__(self) -> None:
        if not 0.0 < self.merge_sse_pixels < self.split_sse_pixels:
            raise ValueError("merge SSE must be smaller than split SSE")
        if not 0 <= self.max_level <= 20:
            raise ValueError("max_level must be in 0..20")
        budgets = (
            self.max_desired_patches,
            self.max_gpu_patches,
            self.build_budget_per_frame,
            self.upload_budget_per_frame,
        )
        if min(budgets) <= 0:
            raise ValueError("terrain budgets must be positive")


class CubeSphereTerrain:
    """World-side terrain system with a renderer-independent frame contract."""

    def __init__(
        self,
        planet: PlanetModel,
        height_source: HeightProvider,
        settings: TerrainSettings,
    ) -> None:
        self.planet = planet
        self.height_provider = height_source
        self.height_source = height_source
        self.settings = settings
        self.selector = MixedLodSelector(
            planet,
            settings.patch_resolution,
            settings.max_level,
            settings.split_sse_pixels,
            settings.merge_sse_pixels,
            settings.max_desired_patches,
            settings.lod_changes_per_update,
        )
        self.tile_manager = TerrainTileManager(
            planet,
            self.selector,
            settings.max_gpu_patches,
            settings.cache_capacity,
            settings.build_budget_per_frame,
            settings.upload_budget_per_frame,
            settings.selection_interval_s,
        )
        self._consumers: list[SurfaceCoverageConsumer] = []
        self._resident = frozenset()

    def add_coverage_consumer(self, consumer: SurfaceCoverageConsumer) -> None:
        self._consumers.append(consumer)

    def update(
        self,
        camera: PlanetCamera,
        viewport_width: int,
        viewport_height: int,
        now: float | None = None,
    ) -> TerrainFrame:
        """Advance streaming and return data-only terrain operations."""

        frame = self.tile_manager.update(
            camera,
            viewport_width,
            viewport_height,
            now,
        )
        self._notify_residency(frame)
        return frame

    def _notify_residency(self, frame: TerrainFrame) -> None:
        added = tuple(sorted(frame.resident - self._resident))
        removed = tuple(sorted(self._resident - frame.resident))
        if added or removed:
            event = PatchResidencyEvent(added, removed)
            for consumer in self._consumers:
                consumer.on_patch_residency_changed(event)
        self._resident = frame.resident

    def select_patches(
        self,
        camera: PlanetCamera,
        viewport_height: int,
    ) -> tuple[PatchKey, ...]:
        return tuple(sorted(self.selector.select(camera, viewport_height)))

    def describe_surface(self, direction_global: Vec3d) -> SurfaceDescriptor:
        return describe_surface(
            self.planet,
            self.height_provider,
            direction_global,
            self.settings.canonical_surface_level,
        )


__all__ = [
    "CubeSphereTerrain",
    "TerrainSettings",
    "PatchResidencyEvent",
    "SurfaceCoverageConsumer",
    "SurfaceDescriptor",
    "TerrainPatchRenderDescriptor",
    "PatchKey",
    "PatchState",
    "PatchUploadRequest",
    "PatchReleaseRequest",
    "TerrainDebugStats",
    "TerrainFrame",
    "ProceduralHeightProvider",
    "DemHeightProvider",
    "cube_face_direction",
    "direction_to_cube_face_uv",
    "surface_cell_id",
]
