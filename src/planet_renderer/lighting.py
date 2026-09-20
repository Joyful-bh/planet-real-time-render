"""Shared radiometric lighting state for every rendering subsystem."""

from dataclasses import dataclass
import math

import numpy as np

from .planet import Vec3d, normalize


@dataclass(frozen=True)
class LightingState:
    sun_direction_global: Vec3d
    sun_angular_radius_degrees: float
    solar_irradiance: tuple[float, float, float]

    def __post_init__(self) -> None:
        direction = normalize(np.asarray(self.sun_direction_global, dtype=np.float64))
        object.__setattr__(self, "sun_direction_global", direction)
        if not 0.01 <= self.sun_angular_radius_degrees <= 5.0:
            raise ValueError("sun angular radius must be in 0.01..5 degrees")
        if len(self.solar_irradiance) != 3 or not all(
            math.isfinite(value) and value >= 0.0
            for value in self.solar_irradiance
        ):
            raise ValueError("solar irradiance must contain three finite values")

    @property
    def sun_solid_angle_sr(self) -> float:
        """Solid angle of the finite solar disk in steradians."""

        radius = math.radians(self.sun_angular_radius_degrees)
        return 2.0 * math.pi * (1.0 - math.cos(radius))

    @property
    def sun_projected_solid_angle_sr(self) -> float:
        """Cosine-weighted solid angle seen by a sun-facing surface."""

        radius = math.radians(self.sun_angular_radius_degrees)
        return math.pi * math.sin(radius) ** 2

    @property
    def sun_disk_radiance(self) -> tuple[float, float, float]:
        """Uniform-disk radiance whose integral equals solar irradiance.

        A distant uniform disk obeys ``E = L * pi * sin(alpha)^2`` for a
        surface facing its centre. Deriving radiance here prevents the visible
        sun and atmospheric scattering from drifting onto unrelated scales.
        """

        projected_solid_angle = self.sun_projected_solid_angle_sr
        return tuple(
            value / projected_solid_angle for value in self.solar_irradiance
        )


@dataclass(frozen=True)
class StaticLightingProvider:
    """The sole M0 lighting producer; celestial replaces it in M5."""

    state: LightingState

    def snapshot(self) -> LightingState:
        return self.state
