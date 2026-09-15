"""双精度行星自由飞行相机。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .planet import LocalFrame, PlanetModel, Vec3d, normalize


@dataclass
class PlanetCamera:
    position_global: Vec3d
    yaw_degrees: float = 0.0
    pitch_degrees: float = -8.0
    vertical_fov_degrees: float = 60.0

    def __post_init__(self) -> None:
        self.position_global = np.asarray(self.position_global, dtype=np.float64).copy()
        self.pitch_degrees = float(np.clip(self.pitch_degrees, -89.0, 89.0))

    def view_basis_local(self) -> tuple[Vec3d, Vec3d, Vec3d]:
        """返回局部 East-Up-North 空间中的 right, view_up, forward。"""
        yaw, pitch = math.radians(self.yaw_degrees), math.radians(self.pitch_degrees)
        horizontal = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float64)
        forward = normalize(
            horizontal * math.cos(pitch) + np.array([0.0, 1.0, 0.0]) * math.sin(pitch)
        )
        right = normalize(np.cross(np.array([0.0, 1.0, 0.0]), forward))
        view_up = normalize(np.cross(forward, right))
        return right, view_up, forward

    def move_local(
        self, planet: PlanetModel, east_m: float, up_m: float, north_m: float
    ) -> None:
        """Move in the local frame while preserving spherical great-circle motion.

        The east/north component is tangent to the sphere at the current
        position.  A linear world-space offset followed by recomputing the
        local frame produces a constant-bearing (rhumb-line) path, which
        spirals toward a pole for most headings.  Instead, the tangent
        component rotates the position around the great-circle normal.  The
        camera forward vector is rotated by the same amount and converted back
        to yaw/pitch so the heading is parallel-transported along that orbit.
        The up component changes only the radial altitude.
        """

        east_m = float(east_m)
        up_m = float(up_m)
        north_m = float(north_m)
        if not np.isfinite([east_m, up_m, north_m]).all():
            raise ValueError("camera movement must contain finite values")

        frame = planet.local_frame(self.position_global)
        _, _, forward_local = self.view_basis_local()
        forward_global = frame.local_to_global_direction(forward_local)

        minimum_radius = planet.radius_m + 0.5
        current_radius = float(np.linalg.norm(self.position_global))
        current_radius = max(current_radius, minimum_radius)
        radial = normalize(self.position_global)

        tangent_local = np.array([east_m, 0.0, north_m], dtype=np.float64)
        tangent_distance = float(np.linalg.norm(tangent_local))
        if tangent_distance > 1.0e-12:
            tangent_global = frame.local_to_global_direction(tangent_local)
            tangent_direction = tangent_global / tangent_distance
            orbit_axis = normalize(np.cross(radial, tangent_direction))
            angle = tangent_distance / current_radius
            radial = self._rotate_about_axis(radial, orbit_axis, angle)
            forward_global = self._rotate_about_axis(forward_global, orbit_axis, angle)

        target_radius = max(minimum_radius, current_radius + up_m)
        self.position_global = radial * target_radius

        if tangent_distance > 1.0e-12:
            new_frame = planet.local_frame(self.position_global)
            local_forward = new_frame.global_to_local_direction(forward_global)
            local_forward = normalize(local_forward)
            self.pitch_degrees = math.degrees(
                math.asin(float(np.clip(local_forward[1], -1.0, 1.0)))
            )
            self.yaw_degrees = math.degrees(
                math.atan2(float(local_forward[0]), float(local_forward[2]))
            )

    @staticmethod
    def _rotate_about_axis(value: Vec3d, axis: Vec3d, angle: float) -> Vec3d:
        """Rotate a vector with Rodrigues' formula around a unit axis."""

        cosine = math.cos(angle)
        sine = math.sin(angle)
        return (
            value * cosine
            + np.cross(axis, value) * sine
            + axis * np.dot(axis, value) * (1.0 - cosine)
        )

    def frame(self, planet: PlanetModel) -> LocalFrame:
        return planet.local_frame(self.position_global)
