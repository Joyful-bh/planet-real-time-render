"""Hierarchical procedural landforms for an Earth-like experimental planet."""

from dataclasses import dataclass

import numpy as np
import taichi as ti

from .planet import Vec3d, normalize


@dataclass(frozen=True)
class LandformsTerrainConfig:
    """Private parameters for ``procedural_landforms_v1``."""

    seed: int = 41
    warp_frequency: float = 2.3
    warp_strength: float = 0.75
    continent_frequency: float = 1.35
    ocean_threshold: float = 0.52
    coast_width: float = 0.075
    ocean_depth_m: float = 4200.0
    continent_relief_m: float = 800.0
    mountain_belt_frequency: float = 3.2
    mountain_belt_start: float = 0.83
    mountain_belt_end: float = 0.97
    mountain_frequency: float = 8.0
    mountain_amplitude_m: float = 3500.0
    plateau_frequency: float = 4.2
    plateau_start: float = 0.55
    plateau_end: float = 0.69
    plateau_height_m: float = 1800.0
    basin_start: float = 0.56
    basin_end: float = 0.72
    basin_depth_m: float = 1150.0
    canyon_frequency: float = 19.0
    canyon_width: float = 0.055
    canyon_depth_m: float = 750.0
    detail_frequency: float = 96.0
    plain_detail_m: float = 180.0
    rugged_detail_m: float = 360.0
    min_height_m: float = -5000.0
    max_height_m: float = 8500.0

    def __post_init__(self) -> None:
        positive = (
            self.warp_frequency,
            self.continent_frequency,
            self.coast_width,
            self.ocean_depth_m,
            self.mountain_belt_frequency,
            self.mountain_frequency,
            self.plateau_frequency,
            self.canyon_frequency,
            self.canyon_width,
            self.detail_frequency,
        )
        if min(positive) <= 0.0:
            raise ValueError("landform frequencies, widths and depths must be positive")
        if not 0.05 <= self.ocean_threshold <= 0.95:
            raise ValueError("ocean_threshold must be in 0.05..0.95")
        threshold_pairs = (
            ("mountain belt", self.mountain_belt_start, self.mountain_belt_end),
            ("plateau", self.plateau_start, self.plateau_end),
            ("basin", self.basin_start, self.basin_end),
        )
        for name, start, end in threshold_pairs:
            if not 0.0 <= start < end <= 1.0:
                raise ValueError(f"{name} thresholds must satisfy 0 <= start < end <= 1")
        if min(
            self.continent_relief_m,
            self.mountain_amplitude_m,
            self.plateau_height_m,
            self.basin_depth_m,
            self.canyon_depth_m,
            self.plain_detail_m,
            self.rugged_detail_m,
        ) < 0.0:
            raise ValueError("landform amplitudes must be non-negative")
        if self.min_height_m >= self.max_height_m:
            raise ValueError("min_height_m must be below max_height_m")


@ti.data_oriented
class LandformsTerrainGenerator:
    """Generate coherent oceans and regionally distinct landform families.

    A sample contains height followed by weights for ocean, plain, mountain,
    plateau, basin and canyon. The weights describe the generator result; the
    renderer later maps them to a small stable material palette.
    """

    supports_gpu = True

    def __init__(self, config: LandformsTerrainConfig) -> None:
        self.config = config
        for name, value in config.__dict__.items():
            setattr(self, name, value)
        self.height_range_m = (config.min_height_m, config.max_height_m)

    @staticmethod
    def _smoothstep_cpu(edge0: float, edge1: float, value: float) -> float:
        t = float(np.clip((value - edge0) / max(edge1 - edge0, 1.0e-12), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

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

    def _ridged_cpu(self, position: Vec3d, seed: int, octaves: int) -> float:
        value = 0.0
        total = 0.0
        amplitude = 0.55
        point = position.copy()
        for octave in range(octaves):
            noise = self._noise_cpu(point, seed + octave * 1297)
            ridge = 1.0 - abs(noise * 2.0 - 1.0)
            value += ridge * ridge * amplitude
            total += amplitude
            point = point * 2.07 + np.array([-5.2, 8.3, 2.7])
            amplitude *= 0.48
        return value / total

    def sample_terrain_m(self, direction_global: Vec3d) -> tuple[float, ...]:
        p = self.config
        direction = normalize(np.asarray(direction_global, np.float64))
        warp = (
            np.array(
                [
                    self._fbm_cpu(direction * p.warp_frequency + 11.0, p.seed + 17, 3),
                    self._fbm_cpu(direction * p.warp_frequency - 7.0, p.seed + 31, 3),
                    self._fbm_cpu(direction * p.warp_frequency + 3.0, p.seed + 47, 3),
                ]
            )
            - 0.5
        )
        warped = direction + warp * p.warp_strength
        continent = self._fbm_cpu(
            warped * p.continent_frequency,
            p.seed + 101,
            5,
        )
        land = self._smoothstep_cpu(
            p.ocean_threshold - p.coast_width,
            p.ocean_threshold + p.coast_width,
            continent,
        )
        above = max(continent - p.ocean_threshold, 0.0) / max(
            1.0 - p.ocean_threshold, 1.0e-6
        )
        below = max(p.ocean_threshold - continent, 0.0) / max(
            p.ocean_threshold, 1.0e-6
        )
        base = (
            above**1.25 * p.continent_relief_m
            - below**0.72 * p.ocean_depth_m
        )

        belt_source = self._fbm_cpu(
            warped * p.mountain_belt_frequency + 4.7,
            p.seed + 211,
            4,
        )
        belt_ridge = 1.0 - abs(belt_source * 2.0 - 1.0)
        mountain = self._smoothstep_cpu(
            p.mountain_belt_start,
            p.mountain_belt_end,
            belt_ridge,
        ) * land

        province = self._fbm_cpu(
            warped * p.plateau_frequency - 9.2,
            p.seed + 307,
            4,
        )
        plateau = self._smoothstep_cpu(
            p.plateau_start,
            p.plateau_end,
            province,
        ) * land * (1.0 - mountain)
        basin = (
            self._smoothstep_cpu(p.basin_start, p.basin_end, 1.0 - province)
            * land
            * (1.0 - mountain)
            * (1.0 - plateau)
        )

        canyon_source = self._fbm_cpu(
            warped * p.canyon_frequency + 2.1,
            p.seed + 401,
            3,
        )
        canyon_distance = abs(canyon_source - 0.5)
        canyon = (
            1.0
            - self._smoothstep_cpu(
                p.canyon_width,
                p.canyon_width * 2.8,
                canyon_distance,
            )
        ) * land * (0.25 + 0.75 * (plateau + basin))

        mountain_detail = self._ridged_cpu(
            warped * p.mountain_frequency,
            p.seed + 503,
            5,
        )
        detail = self._fbm_cpu(
            warped * p.detail_frequency,
            p.seed + 601,
            4,
        )
        height = base
        height += mountain * p.mountain_amplitude_m * (0.18 + 0.82 * mountain_detail)
        height += plateau * p.plateau_height_m * (0.88 + 0.12 * detail)
        height -= basin * p.basin_depth_m * (0.55 + 0.45 * detail)
        height -= canyon * p.canyon_depth_m * (0.45 + 0.55 * mountain_detail)
        rugged = float(np.clip(mountain + plateau * 0.35 + canyon * 0.55, 0.0, 1.0))
        height += (
            (detail - 0.5)
            * 2.0
            * land
            * (p.plain_detail_m * (1.0 - rugged) + p.rugged_detail_m * rugged)
        )
        height = float(np.clip(height, p.min_height_m, p.max_height_m))

        occupied = float(np.clip(max(mountain, plateau, basin, canyon), 0.0, 1.0))
        plain = land * (1.0 - occupied)
        weights = np.array([1.0 - land, plain, mountain, plateau, basin, canyon])
        weights /= max(float(weights.sum()), 1.0e-12)
        return (height, *(float(value) for value in weights))

    def sample_height_m(self, direction_global: Vec3d) -> float:
        return self.sample_terrain_m(direction_global)[0]

    def estimate_error_m(
        self,
        direction_global: Vec3d,
        level: int,
        patch_resolution: int,
        planet_radius_m: float,
    ) -> float:
        """Estimate unresolved local relief for screen-space LOD selection."""

        sample = self.sample_terrain_m(direction_global)
        _, ocean, plain, mountain, plateau, basin, canyon = sample
        samples = max((1 << level) * patch_resolution, 1)

        def unresolved(amplitude: float, frequency: float) -> float:
            ratio = min(6.0 * frequency / samples, 1.0)
            return amplitude * ratio * ratio

        regional = (
            mountain * unresolved(self.mountain_amplitude_m, self.mountain_frequency)
            + plateau * unresolved(self.plateau_height_m, self.plateau_frequency)
            + basin * unresolved(self.basin_depth_m, self.plateau_frequency)
            + canyon * unresolved(self.canyon_depth_m, self.canyon_frequency)
        )
        local_amplitude = plain * self.plain_detail_m + (
            mountain + plateau + canyon
        ) * self.rugged_detail_m
        detail = unresolved(local_amplitude, self.detail_frequency)
        ocean_error = ocean * unresolved(180.0, self.continent_frequency * 3.0)
        return max(regional + detail + ocean_error, 0.25)

    @ti.func
    def _smoothstep_gpu(self, edge0: ti.f32, edge1: ti.f32, value: ti.f32) -> ti.f32:
        t = ti.min(ti.max((value - edge0) / ti.max(edge1 - edge0, 1.0e-8), 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

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
    def _fbm_gpu(self, position: ti.template(), seed: ti.i32, octaves: ti.template()) -> ti.f32:
        value = 0.0
        total = 0.0
        amplitude = 0.5
        point = position
        for octave in ti.static(range(6)):
            if ti.static(octave < octaves):
                value += self._noise_gpu(point, seed + octave * 1013) * amplitude
                total += amplitude
            point = point * 2.03 + ti.Vector([7.1, -3.7, 5.3])
            amplitude *= 0.5
        return value / total

    @ti.func
    def _ridged_gpu(self, position: ti.template(), seed: ti.i32, octaves: ti.template()) -> ti.f32:
        value = 0.0
        total = 0.0
        amplitude = 0.55
        point = position
        for octave in ti.static(range(6)):
            if ti.static(octave < octaves):
                noise = self._noise_gpu(point, seed + octave * 1297)
                ridge = 1.0 - ti.abs(noise * 2.0 - 1.0)
                value += ridge * ridge * amplitude
                total += amplitude
            point = point * 2.07 + ti.Vector([-5.2, 8.3, 2.7])
            amplitude *= 0.48
        return value / total

    @ti.func
    def sample_terrain_gpu(self, direction: ti.template()):
        warp = (
            ti.Vector(
                [
                    self._fbm_gpu(direction * self.warp_frequency + 11.0, self.seed + 17, 3),
                    self._fbm_gpu(direction * self.warp_frequency - 7.0, self.seed + 31, 3),
                    self._fbm_gpu(direction * self.warp_frequency + 3.0, self.seed + 47, 3),
                ]
            )
            - 0.5
        )
        warped = direction + warp * self.warp_strength
        continent = self._fbm_gpu(
            warped * self.continent_frequency,
            self.seed + 101,
            5,
        )
        land = self._smoothstep_gpu(
            self.ocean_threshold - self.coast_width,
            self.ocean_threshold + self.coast_width,
            continent,
        )
        above = ti.max(continent - self.ocean_threshold, 0.0) / ti.max(
            1.0 - self.ocean_threshold, 1.0e-6
        )
        below = ti.max(self.ocean_threshold - continent, 0.0) / ti.max(
            self.ocean_threshold, 1.0e-6
        )
        base = above**1.25 * self.continent_relief_m - below**0.72 * self.ocean_depth_m

        belt_source = self._fbm_gpu(
            warped * self.mountain_belt_frequency + 4.7,
            self.seed + 211,
            4,
        )
        belt_ridge = 1.0 - ti.abs(belt_source * 2.0 - 1.0)
        mountain = self._smoothstep_gpu(
            self.mountain_belt_start,
            self.mountain_belt_end,
            belt_ridge,
        ) * land

        province = self._fbm_gpu(
            warped * self.plateau_frequency - 9.2,
            self.seed + 307,
            4,
        )
        plateau = self._smoothstep_gpu(
            self.plateau_start,
            self.plateau_end,
            province,
        ) * land * (1.0 - mountain)
        basin = (
            self._smoothstep_gpu(self.basin_start, self.basin_end, 1.0 - province)
            * land
            * (1.0 - mountain)
            * (1.0 - plateau)
        )

        canyon_source = self._fbm_gpu(
            warped * self.canyon_frequency + 2.1,
            self.seed + 401,
            3,
        )
        canyon_distance = ti.abs(canyon_source - 0.5)
        canyon = (
            1.0
            - self._smoothstep_gpu(
                self.canyon_width,
                self.canyon_width * 2.8,
                canyon_distance,
            )
        ) * land * (0.25 + 0.75 * (plateau + basin))

        mountain_detail = self._ridged_gpu(
            warped * self.mountain_frequency,
            self.seed + 503,
            5,
        )
        detail = self._fbm_gpu(
            warped * self.detail_frequency,
            self.seed + 601,
            4,
        )
        height = base
        height += mountain * self.mountain_amplitude_m * (0.18 + 0.82 * mountain_detail)
        height += plateau * self.plateau_height_m * (0.88 + 0.12 * detail)
        height -= basin * self.basin_depth_m * (0.55 + 0.45 * detail)
        height -= canyon * self.canyon_depth_m * (0.45 + 0.55 * mountain_detail)
        rugged = ti.min(ti.max(mountain + plateau * 0.35 + canyon * 0.55, 0.0), 1.0)
        height += (
            (detail - 0.5)
            * 2.0
            * land
            * (self.plain_detail_m * (1.0 - rugged) + self.rugged_detail_m * rugged)
        )
        height = ti.min(ti.max(height, self.min_height_m), self.max_height_m)

        occupied = ti.min(ti.max(ti.max(ti.max(mountain, plateau), ti.max(basin, canyon)), 0.0), 1.0)
        plain = land * (1.0 - occupied)
        weights = ti.Vector([1.0 - land, plain, mountain, plateau, basin, canyon])
        weights /= ti.max(weights.sum(), 1.0e-8)
        return ti.Vector(
            [height, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]]
        )

    @ti.func
    def sample_height_gpu(self, direction: ti.template()) -> ti.f32:
        return self.sample_terrain_gpu(direction)[0]


__all__ = ["LandformsTerrainConfig", "LandformsTerrainGenerator"]
