"""Analytic ocean coverage and footprint-filtered surface slopes.

The reference-radius sphere supplies stable planet-scale coverage.  Resolved
wave slopes perturb the shading normal while sub-pixel bands are removed from
the explicit signal and exported as statistical slope variance for the water
BRDF.  This prevents distant waves from turning into a coherent moire pattern.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import taichi as ti


OCEAN_SURFACE_ID = 2_147_000_000
_SPECTRUM_CASCADES = 3
_SPECTRUM_PERIODS_M = (2048.0, 256.0, 32.0)
_SPECTRUM_HEIGHTS_M = (0.62, 0.14, 0.025)
_SPECTRUM_MODE_X = (3, 4, 4, 5, 5, 3, 2, 1, -1, -2, -3, -4)
_SPECTRUM_MODE_Y = (1, 1, 2, 1, 2, 2, 3, 4, 4, 3, 2, 1)
_SPECTRUM_WEIGHTS = (1.0, 0.82, 0.72, 0.60, 0.52, 0.68, 0.55, 0.42, 0.36, 0.31, 0.27, 0.22)
_SPECTRUM_PHASES = (0.31, 1.73, 3.11, 4.67, 0.93, 2.39, 5.21, 3.83, 1.17, 5.72, 2.83, 4.09)


def opaque_surface_height_m(
    terrain_height_m: float,
    ocean_enabled: bool,
) -> float:
    """Return the first opaque radial surface for camera/atmosphere queries."""

    terrain_height_m = float(terrain_height_m)
    return max(terrain_height_m, 0.0) if ocean_enabled else terrain_height_m


@dataclass(frozen=True)
class OceanConfig:
    """Configuration for sea-level coverage and the water material.

    ``surface_enabled`` owns the physical sea-level boundary. ``enabled``
    only selects the advanced wave/BRDF path; disabling it falls back to a
    flat diffuse sea-level surface instead of exposing the below-radius seabed.
    """

    surface_enabled: bool = True
    enabled: bool = True
    albedo: tuple[float, float, float] = (0.012, 0.055, 0.16)
    roughness: float = 0.12
    normal_strength: float = 0.32
    sky_reflection_strength: float = 1.0
    dielectric_f0: float = 0.02
    absorption_per_m: tuple[float, float, float] = (0.14, 0.055, 0.022)
    scattering_per_m: tuple[float, float, float] = (0.0025, 0.005, 0.012)
    max_visible_depth_m: float = 80.0
    spectrum_resolution: int = 128
    wind_speed_mps: float = 8.0
    wind_direction_degrees: float = 35.0
    wave_height_scale: float = 1.0
    choppiness: float = 0.65
    geometry_cascades: int = 2
    geometry_max_distance_m: float = 25_000.0
    refraction_index: float = 1.333
    refraction_strength: float = 1.0
    refraction_max_offset_pixels: float = 48.0

    def __post_init__(self) -> None:
        if len(self.albedo) != 3:
            raise ValueError("ocean.albedo must contain three components")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.albedo):
            raise ValueError("ocean.albedo components must be in 0..1")
        if not 0.02 <= self.roughness <= 1.0:
            raise ValueError("ocean.roughness must be in 0.02..1")
        if not 0.0 <= self.normal_strength <= 1.0:
            raise ValueError("ocean.normal_strength must be in 0..1")
        if not 0.0 <= self.sky_reflection_strength <= 1.0:
            raise ValueError("ocean.sky_reflection_strength must be in 0..1")
        if not 0.0 <= self.dielectric_f0 <= 0.2:
            raise ValueError("ocean.dielectric_f0 must be in 0..0.2")
        for name, values in (
            ("absorption_per_m", self.absorption_per_m),
            ("scattering_per_m", self.scattering_per_m),
        ):
            if len(values) != 3 or any(float(value) < 0.0 for value in values):
                raise ValueError(f"ocean.{name} must contain three non-negative values")
        if self.max_visible_depth_m <= 0.0:
            raise ValueError("ocean.max_visible_depth_m must be positive")
        if self.spectrum_resolution not in (32, 64, 128, 256):
            raise ValueError("ocean.spectrum_resolution must be 32, 64, 128 or 256")
        if not 0.1 <= self.wind_speed_mps <= 40.0:
            raise ValueError("ocean.wind_speed_mps must be in 0.1..40")
        if not math.isfinite(self.wind_direction_degrees):
            raise ValueError("ocean.wind_direction_degrees must be finite")
        if not 0.0 <= self.wave_height_scale <= 4.0:
            raise ValueError("ocean.wave_height_scale must be in 0..4")
        if not 0.0 <= self.choppiness <= 1.0:
            raise ValueError("ocean.choppiness must be in 0..1")
        if not 0 <= self.geometry_cascades <= _SPECTRUM_CASCADES:
            raise ValueError("ocean.geometry_cascades must be in 0..3")
        if self.geometry_max_distance_m < 0.0:
            raise ValueError("ocean.geometry_max_distance_m must be non-negative")
        if not 1.0 <= self.refraction_index <= 2.0:
            raise ValueError("ocean.refraction_index must be in 1..2")
        if not 0.0 <= self.refraction_strength <= 1.0:
            raise ValueError("ocean.refraction_strength must be in 0..1")
        if self.refraction_max_offset_pixels < 0.0:
            raise ValueError(
                "ocean.refraction_max_offset_pixels must be non-negative"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "OceanConfig":
        defaults = cls()
        raw_albedo = values.get("albedo", list(defaults.albedo))
        if not isinstance(raw_albedo, (list, tuple)) or len(raw_albedo) != 3:
            raise ValueError("ocean.albedo must contain three components")
        return cls(
            surface_enabled=bool(
                values.get("surface_enabled", defaults.surface_enabled)
            ),
            enabled=bool(values.get("enabled", defaults.enabled)),
            albedo=tuple(float(value) for value in raw_albedo),
            roughness=float(values.get("roughness", defaults.roughness)),
            normal_strength=float(
                values.get("normal_strength", defaults.normal_strength)
            ),
            sky_reflection_strength=float(
                values.get(
                    "sky_reflection_strength",
                    defaults.sky_reflection_strength,
                )
            ),
            dielectric_f0=float(
                values.get("dielectric_f0", defaults.dielectric_f0)
            ),
            absorption_per_m=tuple(
                float(value)
                for value in values.get(
                    "absorption_per_m",
                    defaults.absorption_per_m,
                )
            ),
            scattering_per_m=tuple(
                float(value)
                for value in values.get(
                    "scattering_per_m",
                    defaults.scattering_per_m,
                )
            ),
            max_visible_depth_m=float(
                values.get("max_visible_depth_m", defaults.max_visible_depth_m)
            ),
            spectrum_resolution=int(
                values.get("spectrum_resolution", defaults.spectrum_resolution)
            ),
            wind_speed_mps=float(
                values.get("wind_speed_mps", defaults.wind_speed_mps)
            ),
            wind_direction_degrees=float(
                values.get(
                    "wind_direction_degrees",
                    defaults.wind_direction_degrees,
                )
            ),
            wave_height_scale=float(
                values.get("wave_height_scale", defaults.wave_height_scale)
            ),
            choppiness=float(values.get("choppiness", defaults.choppiness)),
            geometry_cascades=int(
                values.get("geometry_cascades", defaults.geometry_cascades)
            ),
            geometry_max_distance_m=float(
                values.get(
                    "geometry_max_distance_m",
                    defaults.geometry_max_distance_m,
                )
            ),
            refraction_index=float(
                values.get("refraction_index", defaults.refraction_index)
            ),
            refraction_strength=float(
                values.get("refraction_strength", defaults.refraction_strength)
            ),
            refraction_max_offset_pixels=float(
                values.get(
                    "refraction_max_offset_pixels",
                    defaults.refraction_max_offset_pixels,
                )
            ),
        )


@ti.data_oriented
class OceanRenderer:
    """Rasterize an analytic sea-level sphere into the shared G-buffer."""

    def __init__(self, width: int, height: int, config: OceanConfig):
        self.width = int(width)
        self.height = int(height)
        self.config = config
        spectrum_shape = (
            _SPECTRUM_CASCADES,
            config.spectrum_resolution,
            config.spectrum_resolution,
        )
        self.spectrum_height = ti.field(ti.f32, shape=spectrum_shape)
        self.spectrum_gradient = ti.Vector.field(
            2,
            ti.f32,
            shape=spectrum_shape,
        )
        self.spectrum_displacement = ti.Vector.field(
            2,
            ti.f32,
            shape=spectrum_shape,
        )
        # Row-major derivative of horizontal displacement with respect to
        # the undisturbed spectrum coordinate.  Keeping this next to the
        # spectrum lets shading apply the inverse-map Jacobian cheaply.
        self.spectrum_displacement_jacobian = ti.Vector.field(
            4,
            ti.f32,
            shape=spectrum_shape,
        )
        self._cascade_slope_variance = self._compute_slope_variance()
        self.max_geometry_displacement_m = (
            sum(_SPECTRUM_HEIGHTS_M[: config.geometry_cascades])
            * config.wave_height_scale
            * self._wind_amplitude_scale()
        )

    def _wind_amplitude_scale(self) -> float:
        return min(max(self.config.wind_speed_mps / 8.0, 0.25), 2.0)

    def _compute_slope_variance(self) -> tuple[float, float, float]:
        weight_sum = sum(_SPECTRUM_WEIGHTS)
        amplitude_scale = self.config.wave_height_scale * self._wind_amplitude_scale()
        variances: list[float] = []
        for period, band_height in zip(
            _SPECTRUM_PERIODS_M,
            _SPECTRUM_HEIGHTS_M,
        ):
            variance = 0.0
            for mode_x, mode_y, weight in zip(
                _SPECTRUM_MODE_X,
                _SPECTRUM_MODE_Y,
                _SPECTRUM_WEIGHTS,
            ):
                amplitude = band_height * amplitude_scale * weight / weight_sum
                wave_number = (
                    2.0
                    * math.pi
                    * math.hypot(mode_x, mode_y)
                    / period
                )
                variance += 0.5 * (amplitude * wave_number) ** 2
            variances.append(variance)
        return tuple(variances)  # type: ignore[return-value]

    @ti.kernel
    def _build_directional_spectrum(
        self,
        time_seconds: ti.f32,
        amplitude_scale: ti.f32,
    ):
        """Synthesize three periodic directional bands once per frame.

        The texture is a compact inverse spectral sum.  Rendering cost is
        independent of mode count: surface pixels only perform one bilinear
        lookup per active cascade.  A future FFT/JONSWAP backend can write the
        same height/gradient fields without changing coverage or shading.
        """

        mode_x = ti.Vector(_SPECTRUM_MODE_X)
        mode_y = ti.Vector(_SPECTRUM_MODE_Y)
        weights = ti.Vector(_SPECTRUM_WEIGHTS)
        phases = ti.Vector(_SPECTRUM_PHASES)
        periods = ti.Vector(_SPECTRUM_PERIODS_M)
        band_heights = ti.Vector(_SPECTRUM_HEIGHTS_M)
        weight_sum = ti.static(float(sum(_SPECTRUM_WEIGHTS)))
        resolution = ti.static(self.config.spectrum_resolution)
        for cascade, x, y in self.spectrum_height:
            u = (ti.cast(x, ti.f32) + 0.5) / resolution
            v = (ti.cast(y, ti.f32) + 0.5) / resolution
            height = 0.0
            gradient = ti.Vector.zero(ti.f32, 2)
            displacement = ti.Vector.zero(ti.f32, 2)
            displacement_jacobian = ti.Vector.zero(ti.f32, 4)
            for mode in ti.static(range(len(_SPECTRUM_MODE_X))):
                kx = ti.cast(mode_x[mode], ti.f32)
                ky = ti.cast(mode_y[mode], ti.f32)
                mode_length = ti.sqrt(kx * kx + ky * ky)
                wave_number = 2.0 * math.pi * mode_length / periods[cascade]
                omega = ti.sqrt(9.81 * wave_number)
                phase = (
                    2.0 * math.pi * (kx * u + ky * v)
                    + phases[mode]
                    - omega * time_seconds
                )
                amplitude = (
                    band_heights[cascade]
                    * amplitude_scale
                    * weights[mode]
                    / weight_sum
                )
                height += amplitude * ti.cos(phase)
                derivative = -amplitude * ti.sin(phase) * 2.0 * math.pi
                gradient += ti.Vector(
                    [
                        derivative * kx / periods[cascade],
                        derivative * ky / periods[cascade],
                    ]
                )
                direction = ti.Vector([kx, ky]) / ti.max(mode_length, 1.0e-6)
                wave_vector = ti.Vector(
                    [
                        2.0 * math.pi * kx / periods[cascade],
                        2.0 * math.pi * ky / periods[cascade],
                    ]
                )
                choppy_amplitude = ti.static(self.config.choppiness) * amplitude
                displacement += -choppy_amplitude * ti.sin(phase) * direction
                jacobian_scale = -choppy_amplitude * ti.cos(phase)
                displacement_jacobian += jacobian_scale * ti.Vector(
                    [
                        direction.x * wave_vector.x,
                        direction.x * wave_vector.y,
                        direction.y * wave_vector.x,
                        direction.y * wave_vector.y,
                    ]
                )
            self.spectrum_height[cascade, x, y] = height
            self.spectrum_gradient[cascade, x, y] = gradient
            self.spectrum_displacement[cascade, x, y] = displacement
            self.spectrum_displacement_jacobian[cascade, x, y] = (
                displacement_jacobian
            )

    @ti.func
    def _near_sphere_distance(self, altitude, ray_cosine, radius):
        """Return the cancellation-resistant near hit on the sea-level sphere."""

        origin_radius = radius + altitude
        half_linear = origin_radius * ray_cosine
        constant = altitude * (2.0 * radius + altitude)
        discriminant = half_linear * half_linear - constant
        distance = -1.0
        if discriminant >= 0.0:
            root = ti.sqrt(ti.max(discriminant, 0.0))
            q = -half_linear - ti.select(half_linear >= 0.0, root, -root)
            root0 = -half_linear - root
            root1 = -half_linear + root
            if ti.abs(q) > 1.0e-12:
                root0 = q
                root1 = constant / q
            near = ti.min(root0, root1)
            far = ti.max(root0, root1)
            if near > 0.0:
                distance = near
            elif far > 0.0:
                distance = far
        return distance

    @ti.func
    def _cascade_axes(self, cascade, wind_radians):
        """Return stable global sampling axes, rotated by the shared wind."""

        base_u = ti.Vector([1.0, 0.0, 0.0])
        base_v = ti.Vector([0.0, 0.0, 1.0])
        if cascade == 1:
            base_u = ti.Vector([0.0, 1.0, 0.0])
            base_v = ti.Vector([1.0, 0.0, 0.0])
        elif cascade == 2:
            base_u = ti.Vector([0.0, 0.0, 1.0])
            base_v = ti.Vector([0.0, 1.0, 0.0])
        cosine = ti.cos(wind_radians)
        sine = ti.sin(wind_radians)
        return (
            base_u * cosine + base_v * sine,
            -base_u * sine + base_v * cosine,
        )

    @ti.func
    def _sample_spectrum_texture(self, cascade, uv):
        resolution = ti.static(self.config.spectrum_resolution)
        wrapped = ti.Vector(
            [uv.x - ti.floor(uv.x), uv.y - ti.floor(uv.y)]
        )
        texel = wrapped * resolution - 0.5
        base = ti.cast(ti.floor(texel), ti.i32)
        fraction = texel - ti.cast(base, ti.f32)
        x0 = (base.x % resolution + resolution) % resolution
        y0 = (base.y % resolution + resolution) % resolution
        x1 = (x0 + 1) % resolution
        y1 = (y0 + 1) % resolution
        h00 = self.spectrum_height[cascade, x0, y0]
        h10 = self.spectrum_height[cascade, x1, y0]
        h01 = self.spectrum_height[cascade, x0, y1]
        h11 = self.spectrum_height[cascade, x1, y1]
        g00 = self.spectrum_gradient[cascade, x0, y0]
        g10 = self.spectrum_gradient[cascade, x1, y0]
        g01 = self.spectrum_gradient[cascade, x0, y1]
        g11 = self.spectrum_gradient[cascade, x1, y1]
        d00 = self.spectrum_displacement[cascade, x0, y0]
        d10 = self.spectrum_displacement[cascade, x1, y0]
        d01 = self.spectrum_displacement[cascade, x0, y1]
        d11 = self.spectrum_displacement[cascade, x1, y1]
        j00 = self.spectrum_displacement_jacobian[cascade, x0, y0]
        j10 = self.spectrum_displacement_jacobian[cascade, x1, y0]
        j01 = self.spectrum_displacement_jacobian[cascade, x0, y1]
        j11 = self.spectrum_displacement_jacobian[cascade, x1, y1]
        height0 = h00 * (1.0 - fraction.x) + h10 * fraction.x
        height1 = h01 * (1.0 - fraction.x) + h11 * fraction.x
        gradient0 = g00 * (1.0 - fraction.x) + g10 * fraction.x
        gradient1 = g01 * (1.0 - fraction.x) + g11 * fraction.x
        displacement0 = d00 * (1.0 - fraction.x) + d10 * fraction.x
        displacement1 = d01 * (1.0 - fraction.x) + d11 * fraction.x
        jacobian0 = j00 * (1.0 - fraction.x) + j10 * fraction.x
        jacobian1 = j01 * (1.0 - fraction.x) + j11 * fraction.x
        return (
            height0 * (1.0 - fraction.y) + height1 * fraction.y,
            gradient0 * (1.0 - fraction.y) + gradient1 * fraction.y,
            displacement0 * (1.0 - fraction.y) + displacement1 * fraction.y,
            jacobian0 * (1.0 - fraction.y) + jacobian1 * fraction.y,
        )

    @ti.func
    def _sample_spectrum(
        self,
        global_offset,
        radial,
        phase_offsets: ti.template(),
        wind_radians,
        pixel_footprint,
        geometry_only: ti.template(),
    ):
        height = 0.0
        slope = ti.Vector.zero(ti.f32, 3)
        unresolved_variance = 0.0
        periods = ti.Vector(_SPECTRUM_PERIODS_M)
        variances = ti.Vector(self._cascade_slope_variance)
        min_wavelengths = periods / 5.5
        for cascade in ti.static(range(_SPECTRUM_CASCADES)):
            if ti.static(not geometry_only) or ti.static(
                cascade < self.config.geometry_cascades
            ):
                axis_u, axis_v = self._cascade_axes(cascade, wind_radians)
                base_uv = ti.Vector(
                    [
                        phase_offsets[cascade, 0]
                        + global_offset.dot(axis_u) / periods[cascade],
                        phase_offsets[cascade, 1]
                        + global_offset.dot(axis_v) / periods[cascade],
                    ]
                )
                # Horizontal Gerstner displacement maps an undisturbed
                # coordinate q to x=q+D(q).  Invert that map locally so the
                # implicit radial surface remains a single-valued ray target.
                uv = base_uv
                for _ in ti.static(range(2)):
                    ignored_height, ignored_gradient, band_displacement, ignored_jacobian = self._sample_spectrum_texture(
                        cascade,
                        uv,
                    )
                    uv = base_uv - band_displacement / periods[cascade]
                band_height, band_gradient, ignored_displacement, band_jacobian = self._sample_spectrum_texture(
                    cascade,
                    uv,
                )
                if ti.static(geometry_only):
                    height += band_height
                else:
                    samples_per_wave = min_wavelengths[cascade] / ti.max(
                        pixel_footprint,
                        0.01,
                    )
                    resolved = ti.math.clamp(
                        (samples_per_wave - 2.0) / 4.0,
                        0.0,
                        1.0,
                    )
                    resolved = resolved * resolved * (3.0 - 2.0 * resolved)
                    # dh/dx = (I + dD/dq)^-T dh/dq.  This is what sharpens
                    # crests without inventing extra slope energy or folding
                    # the surface at the configured choppiness range.
                    map00 = 1.0 + band_jacobian.x
                    map01 = band_jacobian.y
                    map10 = band_jacobian.z
                    map11 = 1.0 + band_jacobian.w
                    determinant = map00 * map11 - map01 * map10
                    safe_determinant = determinant
                    if ti.abs(safe_determinant) < 0.15:
                        safe_determinant = ti.select(
                            safe_determinant >= 0.0,
                            0.15,
                            -0.15,
                        )
                    mapped_gradient = ti.Vector(
                        [
                            (map11 * band_gradient.x - map10 * band_gradient.y)
                            / safe_determinant,
                            (-map01 * band_gradient.x + map00 * band_gradient.y)
                            / safe_determinant,
                        ]
                    )
                    tangent_u = axis_u - radial * radial.dot(axis_u)
                    tangent_v = axis_v - radial * radial.dot(axis_v)
                    slope += (
                        tangent_u * mapped_gradient.x
                        + tangent_v * mapped_gradient.y
                    ) * resolved
                    unresolved_variance += variances[cascade] * (
                        1.0 - resolved * resolved
                    )
        return height, slope, unresolved_variance

    @ti.kernel
    def _rasterize(
        self,
        depth: ti.template(),
        position_view: ti.template(),
        normal_global: ti.template(),
        albedo: ti.template(),
        material_weights: ti.template(),
        height_m: ti.template(),
        water_depth_m: ti.template(),
        seabed_albedo: ti.template(),
        seabed_normal: ti.template(),
        slope_variance: ti.template(),
        surface_id: ti.template(),
        surface_cell_id: ti.template(),
        planet_radius: ti.f32,
        camera_altitude: ti.f32,
        east: ti.types.vector(3, ti.f32),
        up: ti.types.vector(3, ti.f32),
        north: ti.types.vector(3, ti.f32),
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
        ocean_albedo: ti.types.vector(3, ti.f32),
        tangent_half_fov: ti.f32,
        phase_offsets: ti.types.ndarray(dtype=ti.f32, ndim=2),
        wind_radians: ti.f32,
        max_displacement: ti.f32,
        geometry_max_distance: ti.f32,
        debug_mode: ti.i32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        origin = ti.Vector([0.0, planet_radius + camera_altitude, 0.0])

        for pixel in ti.grouped(surface_id):
            screen_u = (ti.cast(pixel.x, ti.f32) + 0.5) / self.width
            screen_v = (ti.cast(pixel.y, ti.f32) + 0.5) / self.height
            sx = (screen_u * 2.0 - 1.0) * aspect
            sy = screen_v * 2.0 - 1.0
            ray_view = ti.Vector([sx, sy, 1.0 / tangent_half_fov]).normalized()
            ray_local = (
                right * ray_view.x
                + view_up * ray_view.y
                + forward * ray_view.z
            )
            distance = self._near_sphere_distance(
                camera_altitude,
                ray_local.y,
                planet_radius,
            )
            # Rays which narrowly miss mean sea level can still intersect a
            # displaced crest.  Seed them from the conservative outer sphere.
            if (
                distance <= 0.1
                and max_displacement > 0.0
                and camera_altitude > max_displacement
            ):
                distance = self._near_sphere_distance(
                    camera_altitude - max_displacement,
                    ray_local.y,
                    planet_radius + max_displacement,
                )
            # Fixed-point sphere solves give actual near-field wave geometry
            # without introducing a second triangle/patch pipeline.  Only the
            # two low-frequency bands participate in displacement.
            if distance > 0.1 and distance <= geometry_max_distance:
                for _ in ti.static(range(3)):
                    candidate_local = origin + ray_local * distance
                    candidate_radial_local = candidate_local / ti.max(
                        candidate_local.norm(),
                        1.0,
                    )
                    candidate_radial_global = (
                        east * candidate_radial_local.x
                        + up * candidate_radial_local.y
                        + north * candidate_radial_local.z
                    ).normalized()
                    candidate_global_offset = (
                        east * candidate_local.x
                        + up * (candidate_local.y - planet_radius)
                        + north * candidate_local.z
                    )
                    wave_height, ignored_slope, ignored_variance = self._sample_spectrum(
                        candidate_global_offset,
                        candidate_radial_global,
                        phase_offsets,
                        wind_radians,
                        0.0,
                        True,
                    )
                    displaced_distance = self._near_sphere_distance(
                        camera_altitude - wave_height,
                        ray_local.y,
                        planet_radius + wave_height,
                    )
                    if displaced_distance > 0.1:
                        distance = displaced_distance
            if distance > 0.1:
                ocean_view = ray_view * distance
                nearer = surface_id[pixel] < 0 or ocean_view.z < depth[pixel]
                if nearer:
                    seabed_distance = distance + ti.static(
                        self.config.max_visible_depth_m
                    )
                    seabed_color = ti.Vector([0.025, 0.035, 0.025])
                    seabed_surface_normal = ti.Vector([0.0, 1.0, 0.0])
                    has_seabed = surface_id[pixel] >= 0
                    if has_seabed:
                        seabed_distance = position_view[pixel].norm()
                        seabed_color = albedo[pixel]
                        seabed_surface_normal = normal_global[pixel]
                    optical_depth = ti.math.clamp(
                        seabed_distance - distance,
                        0.0,
                        ti.static(self.config.max_visible_depth_m),
                    )
                    ocean_local = origin + ray_local * distance
                    radial_local = ocean_local / ti.max(ocean_local.norm(), 1.0)
                    radial_global = (
                        east * radial_local.x
                        + up * radial_local.y
                        + north * radial_local.z
                    ).normalized()
                    if not has_seabed:
                        seabed_surface_normal = radial_global
                    global_offset = (
                        east * ocean_local.x
                        + up * (ocean_local.y - planet_radius)
                        + north * ocean_local.z
                    )
                    pixel_footprint = (
                        distance * 2.0 * tangent_half_fov / self.height
                    )
                    resolved_slope = ti.Vector.zero(ti.f32, 3)
                    unresolved_variance = 0.0
                    if ti.static(self.config.enabled):
                        ignored_height, resolved_slope, unresolved_variance = self._sample_spectrum(
                            global_offset,
                            radial_global,
                            phase_offsets,
                            wind_radians,
                            pixel_footprint,
                            False,
                        )
                    water_normal = (
                        radial_global
                        - resolved_slope * ti.static(self.config.normal_strength)
                    ).normalized()
                    color = ocean_albedo
                    if debug_mode == 1:
                        color = ti.Vector([0.02, 0.63, 0.82])
                    elif debug_mode == 2:
                        color = ti.Vector([0.05, 0.18, 0.70])
                    elif debug_mode == 3:
                        color = ti.Vector([0.04, 0.35, 0.62])

                    depth[pixel] = ocean_view.z
                    position_view[pixel] = ocean_view
                    normal_global[pixel] = water_normal
                    albedo[pixel] = color
                    material_weights[pixel] = ti.Vector.zero(ti.f32, 4)
                    height_m[pixel] = ocean_local.norm() - planet_radius
                    water_depth_m[pixel] = optical_depth
                    seabed_albedo[pixel] = seabed_color
                    seabed_normal[pixel] = seabed_surface_normal
                    slope_variance[pixel] = unresolved_variance
                    surface_id[pixel] = OCEAN_SURFACE_ID
                    surface_cell_id[pixel] = -1

    def rasterize(
        self,
        depth,
        position_view,
        normal_global,
        albedo,
        material_weights,
        height_m,
        water_depth_m,
        seabed_albedo,
        seabed_normal,
        slope_variance,
        surface_id,
        surface_cell_id,
        planet_radius_m: float,
        camera_altitude_m: float,
        planet_frame: tuple,
        view_basis: tuple,
        tangent_half_fov: float,
        time_seconds: float,
        debug_mode: int,
    ) -> None:
        if not self.config.surface_enabled:
            return
        east, up, north = planet_frame
        right, view_up, forward = view_basis
        if self.config.enabled:
            self._build_directional_spectrum(
                float(time_seconds),
                float(self.config.wave_height_scale * self._wind_amplitude_scale()),
            )
        wind_radians = math.radians(self.config.wind_direction_degrees)
        sea_anchor = np.asarray(up, dtype=np.float64) * float(planet_radius_m)
        phase_offsets = np.zeros((_SPECTRUM_CASCADES, 2), dtype=np.float32)
        base_axes = (
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )
        cosine = math.cos(wind_radians)
        sine = math.sin(wind_radians)
        for cascade, (raw_u, raw_v) in enumerate(base_axes):
            base_u = np.asarray(raw_u, dtype=np.float64)
            base_v = np.asarray(raw_v, dtype=np.float64)
            axis_u = base_u * cosine + base_v * sine
            axis_v = -base_u * sine + base_v * cosine
            period = _SPECTRUM_PERIODS_M[cascade]
            phase_offsets[cascade, 0] = (sea_anchor.dot(axis_u) / period) % 1.0
            phase_offsets[cascade, 1] = (sea_anchor.dot(axis_v) / period) % 1.0
        self._rasterize(
            depth,
            position_view,
            normal_global,
            albedo,
            material_weights,
            height_m,
            water_depth_m,
            seabed_albedo,
            seabed_normal,
            slope_variance,
            surface_id,
            surface_cell_id,
            float(planet_radius_m),
            float(max(camera_altitude_m, 0.0)),
            tuple(float(value) for value in east),
            tuple(float(value) for value in up),
            tuple(float(value) for value in north),
            tuple(float(value) for value in right),
            tuple(float(value) for value in view_up),
            tuple(float(value) for value in forward),
            self.config.albedo,
            float(tangent_half_fov),
            phase_offsets,
            float(wind_radians),
            float(self.max_geometry_displacement_m if self.config.enabled else 0.0),
            float(self.config.geometry_max_distance_m if self.config.enabled else 0.0),
            int(debug_mode),
        )


__all__ = [
    "OCEAN_SURFACE_ID",
    "OceanConfig",
    "OceanRenderer",
    "opaque_surface_height_m",
]
