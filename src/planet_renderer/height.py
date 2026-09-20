"""Terrain height-model contracts independent of mesh generation and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .planet import Vec3d


class TerrainHeightModel(Protocol):
    """Map a global unit direction to radial height above the reference sphere."""

    supports_gpu: bool
    height_range_m: tuple[float, float]

    def sample_height_m(self, direction_global: Vec3d) -> float: ...

    def estimate_error_m(
        self,
        direction_global: Vec3d,
        level: int,
        patch_resolution: int,
        planet_radius_m: float,
    ) -> float: ...


@dataclass(frozen=True)
class DemTerrainModel:
    """Future DEM height model backed by a tiled pyramid and GPU tile cache."""

    pyramid_uri: str
    supports_gpu: bool = False
    height_range_m: tuple[float, float] = (-11_000.0, 9_000.0)

    def sample_height_m(self, direction_global: Vec3d) -> float:
        raise RuntimeError("DEM terrain model is not connected to a tile pyramid")

    def estimate_error_m(
        self,
        direction_global: Vec3d,
        level: int,
        patch_resolution: int,
        planet_radius_m: float,
    ) -> float:
        span = self.height_range_m[1] - self.height_range_m[0]
        return max(span / max(1 << level, 1), 0.25)


__all__ = ["TerrainHeightModel", "DemTerrainModel"]
