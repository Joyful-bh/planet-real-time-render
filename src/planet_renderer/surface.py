"""Stable surface semantics shared by terrain and future coverage systems.

This module describes *what* is at a point on the planet. It does not create
terrain geometry and does not depend on a renderer or GPU resource.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .height import TerrainHeightModel
from .planet import PlanetModel, Vec3d, normalize
from .terrain_lod import direction_to_cube_face_uv


@dataclass(frozen=True)
class SurfaceDescriptor:
    """LOD-independent data consumed by surface and coverage renderers."""

    cell_id: int
    height_m: float
    normal_global: tuple[float, float, float]
    material_weights: tuple[float, float, float, float]
    seed: int


def surface_cell_id(direction: Vec3d, level: int = 14) -> int:
    """Return a stable cube-sphere cell id independent of render LOD."""

    face, u, v = direction_to_cube_face_uv(direction)
    side = 1 << level
    x = min(max(int((u * 0.5 + 0.5) * side), 0), side - 1)
    y = min(max(int((v * 0.5 + 0.5) * side), 0), side - 1)
    return (face << 28) | (y << 14) | x


def describe_surface(
    planet: PlanetModel,
    height_model: TerrainHeightModel,
    direction_global: Vec3d,
    canonical_level: int = 14,
) -> SurfaceDescriptor:
    """Evaluate height, normal and material weights at a surface direction.

    The finite-difference tangent frame is built from the radial direction, so
    the result remains valid away from a globally-horizontal ground plane.
    """

    direction = normalize(np.asarray(direction_global, dtype=np.float64))
    helper = (
        np.array([0.0, 1.0, 0.0])
        if abs(direction[1]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    tangent_u = normalize(np.cross(helper, direction))
    tangent_v = normalize(np.cross(direction, tangent_u))
    epsilon = 1.0e-5

    def position(sample_direction: Vec3d) -> np.ndarray:
        sample_direction = normalize(sample_direction)
        height = height_model.sample_height_m(sample_direction)
        return sample_direction * (planet.radius_m + height)

    normal = normalize(
        np.cross(
            position(direction + tangent_u * epsilon)
            - position(direction - tangent_u * epsilon),
            position(direction + tangent_v * epsilon)
            - position(direction - tangent_v * epsilon),
        )
    )
    if float(np.dot(normal, direction)) < 0.0:
        normal = -normal

    height = height_model.sample_height_m(direction)
    slope = 1.0 - max(float(np.dot(normal, direction)), 0.0)
    snow = float(np.clip((height - 2600.0) / 1800.0, 0.0, 1.0))
    rock = float(np.clip(slope * 7.0, 0.0, 1.0)) * (1.0 - snow)
    sand = (
        float(np.clip(1.0 - abs(height) / 500.0, 0.0, 1.0))
        * (1.0 - rock)
        * (1.0 - snow)
    )
    weights = np.array(
        [max(1.0 - snow - rock - sand, 0.0), rock, sand, snow],
        dtype=np.float64,
    )
    weights /= max(float(weights.sum()), 1.0e-12)

    cell_id = surface_cell_id(direction, canonical_level)
    return SurfaceDescriptor(
        cell_id=cell_id,
        height_m=height,
        normal_global=tuple(float(value) for value in normal),
        material_weights=tuple(float(value) for value in weights),
        seed=(cell_id * 1_664_525 + 1_013_904_223) & 0x7FFFFFFF,
    )


__all__ = ["SurfaceDescriptor", "surface_cell_id", "describe_surface"]
