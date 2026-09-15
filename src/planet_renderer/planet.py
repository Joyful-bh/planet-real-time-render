"""球形行星坐标、局部切线基与浮动原点。公共长度单位为米。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vec3d = np.ndarray


def _vec3(value: tuple[float, float, float] | Vec3d) -> Vec3d:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("位置和方向必须是三个有限的 float64 数值")
    return result


def normalize(value: Vec3d) -> Vec3d:
    length = float(np.linalg.norm(value))
    if length <= 1.0e-12:
        raise ValueError("不能归一化零向量")
    return value / length


@dataclass(frozen=True)
class LocalFrame:
    """行星表面的 East-Up-North 正交基，列向量位于全局空间。"""

    east: Vec3d
    up: Vec3d
    north: Vec3d

    def local_to_global_direction(self, local: Vec3d) -> Vec3d:
        value = _vec3(local)
        return self.east * value[0] + self.up * value[1] + self.north * value[2]

    def global_to_local_direction(self, global_direction: Vec3d) -> Vec3d:
        value = _vec3(global_direction)
        return np.array(
            [
                np.dot(value, self.east),
                np.dot(value, self.up),
                np.dot(value, self.north),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PlanetModel:
    radius_m: float = 6_360_000.0
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_m) or self.radius_m <= 1.0:
            raise ValueError("planet.radius_m 必须大于 1 米")
        object.__setattr__(
            self, "rotation_axis", tuple(normalize(_vec3(self.rotation_axis)))
        )

    def altitude_m(self, position_global: Vec3d) -> float:
        return float(np.linalg.norm(_vec3(position_global)) - self.radius_m)

    def surface_position(
        self, direction_global: Vec3d, altitude_m: float = 0.0
    ) -> Vec3d:
        return normalize(_vec3(direction_global)) * (self.radius_m + altitude_m)

    def local_frame(self, position_global: Vec3d) -> LocalFrame:
        up = normalize(_vec3(position_global))
        axis = _vec3(self.rotation_axis)
        east = np.cross(axis, up)
        if np.linalg.norm(east) < 1.0e-8:
            fallback = (
                np.array([0.0, 0.0, 1.0])
                if abs(up[2]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
            east = np.cross(fallback, up)
        east = normalize(east)
        north = normalize(np.cross(up, east))
        return LocalFrame(east=east, up=up, north=north)

    def horizon_distance_m(self, altitude_m: float) -> float:
        altitude = max(float(altitude_m), 0.0)
        return math.sqrt(altitude * (2.0 * self.radius_m + altitude))


@dataclass
class FloatingOrigin:
    """CPU 双精度浮动原点；GPU 相对位置由双精度减法后转换。"""

    origin_global: Vec3d
    rebase_threshold_m: float = 10_000.0
    revision: int = 0

    def __post_init__(self) -> None:
        self.origin_global = _vec3(self.origin_global).copy()
        if self.rebase_threshold_m <= 0.0:
            raise ValueError("rebase_threshold_m 必须为正数")

    def update(self, camera_global: Vec3d) -> bool:
        camera = _vec3(camera_global)
        if np.linalg.norm(camera - self.origin_global) < self.rebase_threshold_m:
            return False
        self.origin_global = camera.copy()
        self.revision += 1
        return True

    def relative_f32(self, position_global: Vec3d) -> np.ndarray:
        return (_vec3(position_global) - self.origin_global).astype(np.float32)
