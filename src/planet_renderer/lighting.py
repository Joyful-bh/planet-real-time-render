"""所有渲染子系统共享的静态照明状态契约。"""

from dataclasses import dataclass

import numpy as np

from .planet import Vec3d, normalize


@dataclass(frozen=True)
class LightingState:
    sun_direction_global: Vec3d
    sun_angular_radius_degrees: float
    solar_irradiance: tuple[float, float, float]
    sun_disk_radiance: tuple[float, float, float]

    def __post_init__(self) -> None:
        direction = normalize(np.asarray(self.sun_direction_global, dtype=np.float64))
        object.__setattr__(self, "sun_direction_global", direction)
        if not 0.01 <= self.sun_angular_radius_degrees <= 5.0:
            raise ValueError("太阳角半径超出合法范围")


@dataclass(frozen=True)
class StaticLightingProvider:
    """M0 的唯一照明生产者；M5 将由天体系统替换。"""

    state: LightingState

    def snapshot(self) -> LightingState:
        return self.state

