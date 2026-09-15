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
        frame = planet.local_frame(self.position_global)
        delta = frame.local_to_global_direction(
            np.array([east_m, up_m, north_m], dtype=np.float64)
        )
        self.position_global += delta
        altitude = planet.altitude_m(self.position_global)
        if altitude < 0.5:
            self.position_global = planet.surface_position(self.position_global, 0.5)

    def frame(self, planet: PlanetModel) -> LocalFrame:
        return planet.local_frame(self.position_global)
