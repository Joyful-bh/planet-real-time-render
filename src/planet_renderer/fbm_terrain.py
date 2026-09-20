"""Version-one FBM planet-height algorithm and its private configuration.

Edit :class:`FbmTerrainConfig` for ordinary art direction. When the
terrain formula itself changes, keep the adjacent CPU and GPU composition
methods equivalent. The GPU method is authoritative for rendered geometry.
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti

from .planet import Vec3d, normalize


@dataclass(frozen=True)
class FbmTerrainConfig:
    """Parameters shared by the CPU reference and Taichi GPU evaluators."""

    seed: int = 7
    warp_frequency: float = 3.1
    warp_strength: float = 0.7
    warp_octaves: int = 3
    continent_frequency: float = 1.65
    continent_amplitude_m: float = 2800.0
    continent_octaves: int = 5
    mountain_frequency: float = 8.0
    mountain_amplitude_m: float = 4200.0
    mountain_octaves: int = 5
    ridge_power: float = 3.0
    land_bias_m: float = 700.0
    land_transition_m: float = 1800.0
    min_height_m: float = -5000.0
    max_height_m: float = 8500.0

    def __post_init__(self) -> None:
        octaves = (
            self.warp_octaves,
            self.continent_octaves,
            self.mountain_octaves,
        )
        if min(octaves) < 1 or max(octaves) > 8:
            raise ValueError("procedural terrain octaves must be in 1..8")
        if self.land_transition_m <= 0.0:
            raise ValueError("land_transition_m must be positive")
        if self.min_height_m >= self.max_height_m:
            raise ValueError("min_height_m must be below max_height_m")


@ti.data_oriented
class FbmTerrainGenerator:
    """CPU/GPU implementations of one parameterized procedural terrain recipe."""

    supports_gpu = True

    def __init__(
        self,
        config: FbmTerrainConfig,
    ) -> None:
        self.config = config
        self.seed = config.seed
        self.warp_frequency = config.warp_frequency
        self.warp_strength = config.warp_strength
        self.warp_octaves = config.warp_octaves
        self.continent_frequency = config.continent_frequency
        self.continent_amplitude_m = config.continent_amplitude_m
        self.continent_octaves = config.continent_octaves
        self.mountain_frequency = config.mountain_frequency
        self.mountain_amplitude_m = config.mountain_amplitude_m
        self.mountain_octaves = config.mountain_octaves
        self.ridge_power = config.ridge_power
        self.land_bias_m = config.land_bias_m
        self.land_transition_m = config.land_transition_m
        self.min_height_m = config.min_height_m
        self.max_height_m = config.max_height_m
        self.height_range_m = (config.min_height_m, config.max_height_m)

    @staticmethod
    def _hash3_cpu(x: int, y: int, z: int, seed: int) -> float:
        mask = 0xFFFFFFFF
        value = (
            (x * 0x1F123BB5)
            ^ (y * 0x05491333)
            ^ (z * 0x72E12A4D)
            ^ seed
        ) & mask
        value = ((value ^ (value >> 15)) * 0x2C1B3C6D) & mask
        value = ((value ^ (value >> 12)) * 0x297A2D39) & mask
        value ^= value >> 15
        return (value & mask) / mask

    def _noise_cpu(self, position: Vec3d, seed: int) -> float:
        base = np.floor(position).astype(np.int64)
        fraction = position - base
        weight = fraction * fraction * (3.0 - 2.0 * fraction)
        value = 0.0
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    corner = self._hash3_cpu(
                        int(base[0] + dx),
                        int(base[1] + dy),
                        int(base[2] + dz),
                        seed,
                    )
                    value += (
                        corner
                        * (weight[0] if dx else 1.0 - weight[0])
                        * (weight[1] if dy else 1.0 - weight[1])
                        * (weight[2] if dz else 1.0 - weight[2])
                    )
        return value

    def _fbm_cpu(self, position: Vec3d, seed: int, octaves: int) -> float:
        value = 0.0
        total = 0.0
        amplitude = 0.5
        point = position.copy()
        for octave in range(octaves):
            value += self._noise_cpu(point, seed + octave * 1013) * amplitude
            total += amplitude
            point = point * 2.03 + np.array([7.1, -3.7, 5.3])
            amplitude *= 0.5
        return value / total

    def sample_height_m(self, direction_global: Vec3d) -> float:
        """Evaluate radial height in metres for a global sphere direction."""

        p = self.config
        direction = normalize(np.asarray(direction_global, np.float64))
        warp = (
            np.array(
                [
                    self._fbm_cpu(
                        direction * p.warp_frequency + 11.0,
                        p.seed + 17,
                        p.warp_octaves,
                    ),
                    self._fbm_cpu(
                        direction * p.warp_frequency - 7.0,
                        p.seed + 31,
                        p.warp_octaves,
                    ),
                    self._fbm_cpu(
                        direction * p.warp_frequency + 3.0,
                        p.seed + 47,
                        p.warp_octaves,
                    ),
                ]
            )
            - 0.5
        )
        continent = (
            (
                self._fbm_cpu(
                    direction * p.continent_frequency + warp * p.warp_strength,
                    p.seed,
                    p.continent_octaves,
                )
                - 0.5
            )
            * p.continent_amplitude_m
            * 2.0
        )
        ridge_noise = self._fbm_cpu(
            direction * p.mountain_frequency + warp,
            p.seed + 211,
            p.mountain_octaves,
        )
        ridge = (1.0 - abs(ridge_noise * 2.0 - 1.0)) ** p.ridge_power
        land = np.clip(
            (continent + p.land_bias_m) / p.land_transition_m,
            0.0,
            1.0,
        )
        height = continent + ridge * p.mountain_amplitude_m * land
        return float(np.clip(height, p.min_height_m, p.max_height_m))

    def estimate_error_m(
        self,
        direction_global: Vec3d,
        level: int,
        patch_resolution: int,
        planet_radius_m: float,
    ) -> float:
        samples = max((1 << level) * patch_resolution, 1)
        ratio = min(6.0 * self.mountain_frequency / samples, 1.0)
        return max(self.mountain_amplitude_m * ratio * ratio, 0.25)

    @ti.func
    def _hash3_gpu(self, x: ti.i32, y: ti.i32, z: ti.i32, seed: ti.i32) -> ti.f32:
        value = (
            ti.cast(x, ti.u32) * ti.u32(0x1F123BB5)
            ^ ti.cast(y, ti.u32) * ti.u32(0x05491333)
            ^ ti.cast(z, ti.u32) * ti.u32(0x72E12A4D)
            ^ ti.cast(seed, ti.u32)
        )
        value = (value ^ (value >> 15)) * ti.u32(0x2C1B3C6D)
        value = (value ^ (value >> 12)) * ti.u32(0x297A2D39)
        value = value ^ (value >> 15)
        return ti.cast(value, ti.f32) / 4294967295.0

    @ti.func
    def _noise_gpu(self, position: ti.template(), seed: ti.i32) -> ti.f32:
        base = ti.cast(ti.floor(position), ti.i32)
        fraction = position - ti.cast(base, ti.f32)
        weight = fraction * fraction * (3.0 - 2.0 * fraction)
        value = 0.0
        for dz, dy, dx in ti.static(ti.ndrange(2, 2, 2)):
            value += (
                self._hash3_gpu(base.x + dx, base.y + dy, base.z + dz, seed)
                * (weight.x if dx else 1.0 - weight.x)
                * (weight.y if dy else 1.0 - weight.y)
                * (weight.z if dz else 1.0 - weight.z)
            )
        return value

    @ti.func
    def _fbm_gpu(
        self,
        position: ti.template(),
        seed: ti.i32,
        octaves: ti.template(),
    ) -> ti.f32:
        value = 0.0
        total = 0.0
        amplitude = 0.5
        point = position
        for octave in ti.static(range(8)):
            if ti.static(octave < octaves):
                value += self._noise_gpu(point, seed + octave * 1013) * amplitude
                total += amplitude
            point = point * 2.03 + ti.Vector([7.1, -3.7, 5.3])
            amplitude *= 0.5
        return value / total

    @ti.func
    def sample_height_gpu(self, direction: ti.template()) -> ti.f32:
        """Taichi evaluator used by terrain mesh generation kernels."""

        warp = (
            ti.Vector(
                [
                    self._fbm_gpu(
                        direction * self.warp_frequency + 11.0,
                        self.seed + 17,
                        self.warp_octaves,
                    ),
                    self._fbm_gpu(
                        direction * self.warp_frequency - 7.0,
                        self.seed + 31,
                        self.warp_octaves,
                    ),
                    self._fbm_gpu(
                        direction * self.warp_frequency + 3.0,
                        self.seed + 47,
                        self.warp_octaves,
                    ),
                ]
            )
            - 0.5
        )
        continent = (
            (
                self._fbm_gpu(
                    direction * self.continent_frequency + warp * self.warp_strength,
                    self.seed,
                    self.continent_octaves,
                )
                - 0.5
            )
            * self.continent_amplitude_m
            * 2.0
        )
        ridge_noise = self._fbm_gpu(
            direction * self.mountain_frequency + warp,
            self.seed + 211,
            self.mountain_octaves,
        )
        ridge = (1.0 - ti.abs(ridge_noise * 2.0 - 1.0)) ** self.ridge_power
        land = ti.min(
            ti.max(
                (continent + self.land_bias_m) / self.land_transition_m,
                0.0,
            ),
            1.0,
        )
        height = continent + ridge * self.mountain_amplitude_m * land
        return ti.min(ti.max(height, self.min_height_m), self.max_height_m)

    @ti.func
    def sample_terrain_gpu(self, direction: ti.template()):
        height = self.sample_height_gpu(direction)
        ocean = self._smoothstep_height_gpu(-80.0, -350.0, height)
        mountain = self._smoothstep_height_gpu(1200.0, 3600.0, height)
        plateau = self._smoothstep_height_gpu(500.0, 1900.0, height) * (
            1.0 - mountain
        )
        basin = 0.0
        canyon = 0.0
        plain = ti.max(1.0 - ocean - mountain - plateau, 0.0)
        weights = ti.Vector([ocean, plain, mountain, plateau, basin, canyon])
        weights /= ti.max(weights.sum(), 1.0e-8)
        return ti.Vector(
            [
                height,
                weights[0],
                weights[1],
                weights[2],
                weights[3],
                weights[4],
                weights[5],
            ]
        )

    @ti.func
    def _smoothstep_height_gpu(
        self,
        edge0: ti.f32,
        edge1: ti.f32,
        value: ti.f32,
    ) -> ti.f32:
        t = 0.0
        if edge1 >= edge0:
            t = (value - edge0) / ti.max(edge1 - edge0, 1.0e-8)
        else:
            t = (edge0 - value) / ti.max(edge0 - edge1, 1.0e-8)
        t = ti.min(ti.max(t, 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)


__all__ = ["FbmTerrainConfig", "FbmTerrainGenerator"]
