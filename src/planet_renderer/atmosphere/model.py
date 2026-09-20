"""Double-precision reference mathematics for a spherical atmosphere.

This module is intentionally independent from Taichi.  It is the validation
oracle for shell intersections, density profiles, optical depth and direct
single scattering; the realtime renderer evaluates the same model through
GPU LUTs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import AtmosphereConfig


@dataclass(frozen=True)
class RayInterval:
    start_m: float
    end_m: float
    ends_at_ground: bool = False

    @property
    def length_m(self) -> float:
        return max(self.end_m - self.start_m, 0.0)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or length <= 1.0e-15:
        raise ValueError("atmosphere directions must be finite non-zero vectors")
    return vector / length


def ray_sphere_roots(
    origin: np.ndarray,
    direction: np.ndarray,
    radius_m: float,
) -> tuple[float, float] | None:
    """Return ordered ray parameters for a planet-centred sphere."""

    point = np.asarray(origin, dtype=np.float64)
    ray = _unit(direction)
    projected = float(np.dot(point, ray))
    discriminant = projected * projected - (
        float(np.dot(point, point)) - radius_m * radius_m
    )
    tolerance = 1.0e-12 * max(radius_m * radius_m, 1.0)
    if discriminant < -tolerance:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    return -projected - root, -projected + root


class AtmosphereModel:
    """CPU reference model using planet-centred float64 coordinates."""

    def __init__(self, planet_radius_m: float, config: AtmosphereConfig):
        if not math.isfinite(planet_radius_m) or planet_radius_m <= 1.0:
            raise ValueError("planet radius must be positive and finite")
        self.bottom_radius_m = float(planet_radius_m)
        self.top_radius_m = self.bottom_radius_m + config.top_altitude_m
        self.config = config

    def density(self, altitude_m: float) -> tuple[float, float, float]:
        h = float(np.clip(altitude_m, 0.0, self.config.top_altitude_m))
        rayleigh = math.exp(-h / self.config.rayleigh_scale_height_m)
        mie = math.exp(-h / self.config.mie_scale_height_m)
        absorption = max(
            1.0
            - abs(h - self.config.absorption_peak_altitude_m)
            / self.config.absorption_half_width_m,
            0.0,
        )
        if altitude_m < 0.0 or altitude_m > self.config.top_altitude_m:
            return 0.0, 0.0, 0.0
        return rayleigh, mie, absorption

    def extinction(self, altitude_m: float) -> np.ndarray:
        rayleigh, mie, absorption = self.density(altitude_m)
        return (
            np.asarray(self.config.rayleigh_scattering_per_m) * rayleigh
            + np.asarray(self.config.mie_extinction_per_m) * mie
            + np.asarray(self.config.absorption_extinction_per_m) * absorption
        )

    def atmosphere_interval(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        maximum_distance_m: float = math.inf,
    ) -> RayInterval | None:
        point = np.asarray(origin, dtype=np.float64)
        ray = _unit(direction)
        outer = ray_sphere_roots(point, ray, self.top_radius_m)
        if outer is None or outer[1] <= 0.0:
            return None
        start = max(outer[0], 0.0)
        end = min(outer[1], float(maximum_distance_m))
        ends_at_ground = False
        ground = ray_sphere_roots(point, ray, self.bottom_radius_m)
        if ground is not None:
            boundary_epsilon = max(1.0e-6, self.bottom_radius_m * 1.0e-12)
            candidates = [
                distance
                for distance in ground
                if distance > start + boundary_epsilon
            ]
            if candidates:
                nearest = min(candidates)
                if nearest < end:
                    end = nearest
                    ends_at_ground = True
        if end <= start:
            return None
        return RayInterval(start, end, ends_at_ground)

    def transmittance(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        maximum_distance_m: float = math.inf,
        steps: int = 128,
        ground_is_opaque: bool = False,
    ) -> np.ndarray:
        interval = self.atmosphere_interval(origin, direction, maximum_distance_m)
        if interval is None:
            return np.ones(3, dtype=np.float64)
        if ground_is_opaque and interval.ends_at_ground:
            return np.zeros(3, dtype=np.float64)
        ray = _unit(direction)
        count = max(int(steps), 1)
        step = interval.length_m / count
        optical_depth = np.zeros(3, dtype=np.float64)
        point = np.asarray(origin, dtype=np.float64)
        for index in range(count):
            distance = interval.start_m + (index + 0.5) * step
            radius = float(np.linalg.norm(point + ray * distance))
            optical_depth += self.extinction(radius - self.bottom_radius_m) * step
        return np.exp(-np.minimum(optical_depth, 80.0))

    @staticmethod
    def rayleigh_phase(cosine: float) -> float:
        return 3.0 * (1.0 + cosine * cosine) / (16.0 * math.pi)

    def mie_phase(self, cosine: float) -> float:
        g = self.config.mie_phase_g
        denominator = max(1.0 + g * g - 2.0 * g * cosine, 1.0e-8)
        return (
            3.0
            * (1.0 - g * g)
            * (1.0 + cosine * cosine)
            / (8.0 * math.pi * (2.0 + g * g) * denominator**1.5)
        )

    def single_scattering(
        self,
        origin: np.ndarray,
        view_direction: np.ndarray,
        sun_direction: np.ndarray,
        solar_irradiance: tuple[float, float, float],
        steps: int = 64,
        sun_steps: int = 64,
    ) -> np.ndarray:
        """Reference direct single scattering along the visible shell segment."""

        interval = self.atmosphere_interval(origin, view_direction)
        if interval is None:
            return np.zeros(3, dtype=np.float64)
        view = _unit(view_direction)
        sun = _unit(sun_direction)
        point0 = np.asarray(origin, dtype=np.float64)
        count = max(int(steps), 1)
        step = interval.length_m / count
        view_transmittance = np.ones(3, dtype=np.float64)
        radiance = np.zeros(3, dtype=np.float64)
        cosine = float(np.clip(np.dot(view, sun), -1.0, 1.0))
        phase_r = self.rayleigh_phase(cosine)
        phase_m = self.mie_phase(cosine)
        irradiance = np.asarray(solar_irradiance, dtype=np.float64)

        for index in range(count):
            distance = interval.start_m + (index + 0.5) * step
            point = point0 + view * distance
            altitude = float(np.linalg.norm(point) - self.bottom_radius_m)
            density_r, density_m, _ = self.density(altitude)
            sun_transmittance = self.transmittance(
                point,
                sun,
                steps=sun_steps,
                ground_is_opaque=True,
            )
            source = irradiance * sun_transmittance * (
                np.asarray(self.config.rayleigh_scattering_per_m)
                * density_r
                * phase_r
                + np.asarray(self.config.mie_scattering_per_m)
                * density_m
                * phase_m
            )
            segment_extinction = self.extinction(altitude)
            segment_transmittance = np.exp(-np.minimum(segment_extinction * step, 80.0))
            integral = np.where(
                segment_extinction > 1.0e-12,
                (1.0 - segment_transmittance) / segment_extinction,
                step,
            )
            radiance += view_transmittance * source * integral
            view_transmittance *= segment_transmittance
        return np.maximum(radiance, 0.0)
