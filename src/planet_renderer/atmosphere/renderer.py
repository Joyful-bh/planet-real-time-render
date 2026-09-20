"""Taichi LUT generation and full-screen composition for the atmosphere."""

import math

import numpy as np
import taichi as ti

from .config import AtmosphereConfig
from .diagnostics import AtmosphereDiagnosticView


_PI = math.pi
_LARGE_DISTANCE = 1.0e30
_DIAGNOSTIC_COMPOSITE = int(AtmosphereDiagnosticView.COMPOSITE)
_DIAGNOSTIC_SKY_VIEW = int(AtmosphereDiagnosticView.SKY_VIEW)
_DIAGNOSTIC_CAMERA_TRANSMITTANCE = int(
    AtmosphereDiagnosticView.CAMERA_TRANSMITTANCE
)
_DIAGNOSTIC_TRANSMITTANCE_LUT = int(
    AtmosphereDiagnosticView.TRANSMITTANCE_LUT
)
_DIAGNOSTIC_MULTI_SCATTERING_LUT = int(
    AtmosphereDiagnosticView.MULTI_SCATTERING_LUT
)
_DIAGNOSTIC_AERIAL_SCATTERING = int(
    AtmosphereDiagnosticView.AERIAL_SCATTERING
)
_DIAGNOSTIC_AERIAL_TRANSMITTANCE = int(
    AtmosphereDiagnosticView.AERIAL_TRANSMITTANCE
)


@ti.data_oriented
class AtmosphereRenderer:
    """Render a spherical Rayleigh/Mie atmosphere through compact GPU LUTs.

    LUT geometry is planet-centred, while final composition reconstructs rays
    in the camera's local East-Up-North frame.  The camera remains at the
    floating origin and the planet centre is therefore `(0, -camera_radius, 0)`.
    """

    def __init__(self, width: int, height: int, config: AtmosphereConfig):
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.transmittance_lut = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(
                config.transmittance_lut_width,
                config.transmittance_lut_height,
            ),
        )
        self.multi_scattering_lut = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(
                config.multi_scattering_lut_width,
                config.multi_scattering_lut_height,
            ),
        )
        self.sky_view_lut = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(config.sky_view_lut_width, config.sky_view_lut_height),
        )
        aerial_shape = (
            config.aerial_lut_width,
            config.aerial_lut_height,
            config.aerial_lut_depth,
        )
        self.aerial_scattering_lut = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=aerial_shape,
        )
        frame_shape = (self.width, self.height)
        self.surface_sun_transmittance = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=frame_shape,
        )
        self.surface_sky_radiance = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=frame_shape,
        )
        self._planet_radius_key: float | None = None
        self._multi_scattering_key: tuple[
            float,
            tuple[float, float, float],
            float,
        ] | None = None
        self._sky_key: tuple[
            float,
            float,
            tuple[float, float, float, float],
        ] | None = None
        self.luts_frozen = False
        self.sky_snapshot_altitude_m: float | None = None
        self.sky_snapshot_sun_cosine: float | None = None
        self.aerial_snapshot_altitude_m: float | None = None
        self._aerial_available = False
        self.transmittance_rebuilds = 0
        self.multi_scattering_rebuilds = 0
        self.sky_view_rebuilds = 0
        self.aerial_rebuilds = 0

    @ti.func
    def _rayleigh_beta(self):
        return ti.Vector(
            [
                ti.static(self.config.rayleigh_scattering_per_m[0]),
                ti.static(self.config.rayleigh_scattering_per_m[1]),
                ti.static(self.config.rayleigh_scattering_per_m[2]),
            ]
        )

    @ti.func
    def _mie_scattering_beta(self):
        return ti.Vector(
            [
                ti.static(self.config.mie_scattering_per_m[0]),
                ti.static(self.config.mie_scattering_per_m[1]),
                ti.static(self.config.mie_scattering_per_m[2]),
            ]
        )

    @ti.func
    def _mie_extinction_beta(self):
        return ti.Vector(
            [
                ti.static(self.config.mie_extinction_per_m[0]),
                ti.static(self.config.mie_extinction_per_m[1]),
                ti.static(self.config.mie_extinction_per_m[2]),
            ]
        )

    @ti.func
    def _absorption_beta(self):
        return ti.Vector(
            [
                ti.static(self.config.absorption_extinction_per_m[0]),
                ti.static(self.config.absorption_extinction_per_m[1]),
                ti.static(self.config.absorption_extinction_per_m[2]),
            ]
        )

    @ti.func
    def _ground_albedo(self):
        return ti.Vector(
            [
                ti.static(self.config.ground_albedo[0]),
                ti.static(self.config.ground_albedo[1]),
                ti.static(self.config.ground_albedo[2]),
            ]
        )

    @ti.func
    def _density(self, altitude_m):
        inside = 0.0 <= altitude_m and altitude_m <= ti.static(
            self.config.top_altitude_m
        )
        rayleigh = 0.0
        mie = 0.0
        absorption = 0.0
        if inside:
            rayleigh = ti.exp(
                -altitude_m / ti.static(self.config.rayleigh_scale_height_m)
            )
            mie = ti.exp(
                -altitude_m / ti.static(self.config.mie_scale_height_m)
            )
            absorption = ti.max(
                1.0
                - ti.abs(
                    altitude_m
                    - ti.static(self.config.absorption_peak_altitude_m)
                )
                / ti.static(self.config.absorption_half_width_m),
                0.0,
            )
        return ti.Vector([rayleigh, mie, absorption])

    @ti.func
    def _extinction(self, altitude_m):
        density = self._density(altitude_m)
        return (
            self._rayleigh_beta() * density.x
            + self._mie_extinction_beta() * density.y
            + self._absorption_beta() * density.z
        )

    @ti.func
    def _scattering(self, altitude_m):
        density = self._density(altitude_m)
        return (
            self._rayleigh_beta() * density.x
            + self._mie_scattering_beta() * density.y
        )

    @ti.func
    def _sphere_roots(self, origin: ti.template(), ray: ti.template(), radius):
        projected = origin.dot(ray)
        perpendicular = (origin - ray * projected).norm()
        clearance = radius - perpendicular
        hit = clearance >= 0.0
        near = ti.cast(_LARGE_DISTANCE, ti.f32)
        far = -ti.cast(_LARGE_DISTANCE, ti.f32)
        if hit:
            # Factoring r^2 - d^2 as (r - d)(r + d) avoids subtracting
            # Earth-scale squared values at grazing incidence.
            discriminant = ti.max(clearance, 0.0) * (radius + perpendicular)
            root = ti.sqrt(ti.max(discriminant, 0.0))
            near = -projected - root
            far = -projected + root
        return near, far, hit

    @ti.func
    def _atmosphere_segment(
        self,
        origin: ti.template(),
        ray: ti.template(),
        bottom_radius,
    ):
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        outer_near, outer_far, outer_hit = self._sphere_roots(
            origin,
            ray,
            top_radius,
        )
        start = ti.max(outer_near, 0.0)
        end = outer_far
        valid = outer_hit and end > start
        ends_at_ground = False
        ground_near, ground_far, ground_hit = self._sphere_roots(
            origin,
            ray,
            bottom_radius,
        )
        if valid and ground_hit:
            ground_distance = ti.cast(_LARGE_DISTANCE, ti.f32)
            boundary_epsilon = ti.max(0.001, bottom_radius * 1.0e-8)
            if ground_near > start + boundary_epsilon:
                ground_distance = ground_near
            elif ground_far > start + boundary_epsilon:
                ground_distance = ground_far
            if ground_distance < end:
                end = ground_distance
                ends_at_ground = True
        return start, end, valid, ends_at_ground

    @ti.func
    def _aerial_prefix_segment(
        self,
        origin: ti.template(),
        ray: ti.template(),
        bottom_radius,
    ):
        """Return a horizon-continuous interval for cumulative aerial data.

        A ground-hitting ray ends at its first ground intersection.  A grazing
        ray that misses the reference sphere ends at closest approach instead
        of the far atmosphere exit.  The two endpoints converge at tangency,
        so neighboring froxels retain compatible normalized depth coordinates.
        Outward rays, whose closest point is behind the camera, still end at
        the atmosphere exit.
        """

        start, end, valid, ends_at_ground = self._atmosphere_segment(
            origin,
            ray,
            bottom_radius,
        )
        if valid and not ends_at_ground:
            closest = -origin.dot(ray)
            if closest > start and closest < end:
                end = closest
        valid = valid and end > start
        return start, end, valid

    @ti.func
    def _ray_intersects_ground(self, radius, cosine, bottom_radius):
        radius = ti.max(radius, bottom_radius)
        discriminant = (
            radius * radius * (cosine * cosine - 1.0)
            + bottom_radius * bottom_radius
        )
        return cosine < 0.0 and discriminant >= 0.0

    @ti.func
    def _distance_to_top_atmosphere(self, radius, cosine, bottom_radius):
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        discriminant = (
            radius * radius * (cosine * cosine - 1.0)
            + top_radius * top_radius
        )
        return ti.max(
            -radius * cosine + ti.sqrt(ti.max(discriminant, 0.0)),
            0.0,
        )

    @ti.func
    def _transmittance_uv_to_ray(self, u, v, bottom_radius):
        """Decode the distance-based spherical transmittance parameterization."""

        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        shell_horizon = ti.sqrt(
            ti.max(top_radius * top_radius - bottom_radius * bottom_radius, 0.0)
        )
        rho = shell_horizon * ti.math.clamp(v, 0.0, 1.0)
        radius = ti.sqrt(rho * rho + bottom_radius * bottom_radius)
        distance_min = top_radius - radius
        distance_max = rho + shell_horizon
        distance = distance_min + ti.math.clamp(u, 0.0, 1.0) * (
            distance_max - distance_min
        )
        cosine = 1.0
        if distance > 1.0e-6:
            cosine = (
                top_radius * top_radius
                - radius * radius
                - distance * distance
            ) / (2.0 * radius * distance)
        return radius, ti.math.clamp(cosine, -1.0, 1.0), distance

    @ti.func
    def _transmittance_ray_to_uv(self, radius, cosine, bottom_radius):
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        radius = ti.math.clamp(radius, bottom_radius, top_radius)
        shell_horizon = ti.sqrt(
            ti.max(top_radius * top_radius - bottom_radius * bottom_radius, 0.0)
        )
        rho = ti.sqrt(
            ti.max(radius * radius - bottom_radius * bottom_radius, 0.0)
        )
        distance = self._distance_to_top_atmosphere(
            radius,
            cosine,
            bottom_radius,
        )
        distance_min = top_radius - radius
        distance_max = rho + shell_horizon
        u = (distance - distance_min) / ti.max(
            distance_max - distance_min,
            1.0e-6,
        )
        v = rho / ti.max(shell_horizon, 1.0e-6)
        return ti.math.clamp(u, 0.0, 1.0), ti.math.clamp(v, 0.0, 1.0)

    @ti.func
    def _warped_interval_boundary(
        self,
        origin: ti.template(),
        ray: ti.template(),
        start,
        end,
        fraction,
    ):
        """Place integration boundaries densely around minimum altitude."""

        closest = ti.math.clamp(-origin.dot(ray), start, end)
        distance = start
        epsilon = ti.max((end - start) * 1.0e-6, 1.0e-5)
        if closest <= start + epsilon:
            distance = start + (end - start) * fraction * fraction
        elif closest >= end - epsilon:
            inverse = 1.0 - fraction
            distance = end - (end - start) * inverse * inverse
        elif fraction < 0.5:
            local = fraction * 2.0
            inverse = 1.0 - local
            distance = closest - (closest - start) * inverse * inverse
        else:
            local = fraction * 2.0 - 1.0
            distance = closest + (end - closest) * local * local
        return distance

    @ti.func
    def _sky_horizon_zenith(self, camera_radius, bottom_radius):
        radius = ti.max(camera_radius, bottom_radius + 1.0e-3)
        tangent_cosine = ti.sqrt(
            ti.max(radius * radius - bottom_radius * bottom_radius, 0.0)
        ) / radius
        return _PI - ti.acos(ti.math.clamp(tangent_cosine, 0.0, 1.0))

    @ti.func
    def _sky_v_to_zenith(self, v, camera_radius, bottom_radius):
        """Decode a coordinate concentrated on both sides of the horizon."""

        horizon = self._sky_horizon_zenith(camera_radius, bottom_radius)
        zenith = 0.0
        if v < 0.5:
            inverse = 1.0 - 2.0 * v
            zenith = horizon * (1.0 - inverse * inverse)
        else:
            local = 2.0 * v - 1.0
            zenith = horizon + (_PI - horizon) * local * local
        return zenith

    @ti.func
    def _sky_zenith_to_v(self, zenith, camera_radius, bottom_radius):
        horizon = self._sky_horizon_zenith(camera_radius, bottom_radius)
        v = 0.5
        if zenith < horizon:
            normalized = ti.math.clamp(zenith / ti.max(horizon, 1.0e-6), 0.0, 1.0)
            v = 0.5 * (1.0 - ti.sqrt(ti.max(1.0 - normalized, 0.0)))
        else:
            normalized = ti.math.clamp(
                (zenith - horizon) / ti.max(_PI - horizon, 1.0e-6),
                0.0,
                1.0,
            )
            v = 0.5 + 0.5 * ti.sqrt(normalized)
        return v

    @ti.func
    def _sun_u_to_cosine(self, u, radius, bottom_radius):
        """Decode a solar coordinate concentrated around the local horizon."""

        radius = ti.max(radius, bottom_radius)
        horizon_cosine = -ti.sqrt(
            ti.max(
                1.0 - bottom_radius * bottom_radius / (radius * radius),
                0.0,
            )
        )
        cosine = horizon_cosine
        if u < 0.5:
            inverse = 1.0 - 2.0 * u
            cosine = horizon_cosine - (
                horizon_cosine + 1.0
            ) * inverse * inverse
        else:
            local = 2.0 * u - 1.0
            cosine = horizon_cosine + (
                1.0 - horizon_cosine
            ) * local * local
        return ti.math.clamp(cosine, -1.0, 1.0)

    @ti.func
    def _sun_cosine_to_u(self, cosine, radius, bottom_radius):
        radius = ti.max(radius, bottom_radius)
        horizon_cosine = -ti.sqrt(
            ti.max(
                1.0 - bottom_radius * bottom_radius / (radius * radius),
                0.0,
            )
        )
        u = 0.5
        if cosine < horizon_cosine:
            normalized = ti.math.clamp(
                (horizon_cosine - cosine)
                / ti.max(horizon_cosine + 1.0, 1.0e-6),
                0.0,
                1.0,
            )
            u = 0.5 * (1.0 - ti.sqrt(normalized))
        else:
            normalized = ti.math.clamp(
                (cosine - horizon_cosine)
                / ti.max(1.0 - horizon_cosine, 1.0e-6),
                0.0,
                1.0,
            )
            u = 0.5 + 0.5 * ti.sqrt(normalized)
        return u

    @ti.func
    def _integrate_extinction(
        self,
        origin: ti.template(),
        ray: ti.template(),
        start,
        end,
        bottom_radius,
    ):
        optical_depth = ti.Vector.zero(ti.f32, 3)
        index = 0
        while index < ti.static(self.config.transmittance_steps):
            fraction0 = ti.cast(index, ti.f32) / ti.static(
                self.config.transmittance_steps
            )
            fraction1 = ti.cast(index + 1, ti.f32) / ti.static(
                self.config.transmittance_steps
            )
            distance0 = self._warped_interval_boundary(
                origin,
                ray,
                start,
                end,
                fraction0,
            )
            distance1 = self._warped_interval_boundary(
                origin,
                ray,
                start,
                end,
                fraction1,
            )
            step = distance1 - distance0
            distance = (distance0 + distance1) * 0.5
            altitude = (origin + ray * distance).norm() - bottom_radius
            optical_depth += self._extinction(altitude) * step
            index += 1
        return ti.exp(-ti.min(optical_depth, 80.0))

    @ti.func
    def _sample_transmittance_unoccluded(self, radius, cosine, bottom):
        """Sample optical transmittance to the top boundary without occlusion."""

        sample_radius = ti.max(radius, bottom)
        u, v = self._transmittance_ray_to_uv(
            sample_radius,
            cosine,
            bottom,
        )
        x = ti.math.clamp(
            u * ti.static(self.config.transmittance_lut_width) - 0.5,
            0.0,
            ti.static(self.config.transmittance_lut_width - 1),
        )
        y = ti.math.clamp(
            v * ti.static(self.config.transmittance_lut_height) - 0.5,
            0.0,
            ti.static(self.config.transmittance_lut_height - 1),
        )
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(x0 + 1, ti.static(self.config.transmittance_lut_width - 1))
        y1 = ti.min(y0 + 1, ti.static(self.config.transmittance_lut_height - 1))
        fx = x - ti.cast(x0, ti.f32)
        fy = y - ti.cast(y0, ti.f32)
        low = self.transmittance_lut[x0, y0] * (1.0 - fx) + self.transmittance_lut[
            x1, y0
        ] * fx
        high = self.transmittance_lut[x0, y1] * (1.0 - fx) + self.transmittance_lut[
            x1, y1
        ] * fx
        return low * (1.0 - fy) + high * fy

    @ti.func
    def _sample_transmittance(self, radius, cosine, bottom):
        sample_radius = ti.max(radius, bottom)
        blocked = self._ray_intersects_ground(sample_radius, cosine, bottom)
        result = self._sample_transmittance_unoccluded(
            sample_radius,
            cosine,
            bottom,
        )
        if blocked:
            result = ti.Vector.zero(ti.f32, 3)
        return result

    @ti.func
    def _sample_multi_scattering(self, radius, sun_cosine, bottom):
        altitude_fraction = ti.math.clamp(
            (radius - bottom) / ti.static(self.config.top_altitude_m),
            0.0,
            1.0,
        )
        sample_radius = ti.max(radius, bottom)
        u = self._sun_cosine_to_u(
            sun_cosine,
            sample_radius,
            bottom,
        )
        v = ti.sqrt(altitude_fraction)
        x = ti.math.clamp(
            u * ti.static(self.config.multi_scattering_lut_width) - 0.5,
            0.0,
            ti.static(self.config.multi_scattering_lut_width - 1),
        )
        y = ti.math.clamp(
            v * ti.static(self.config.multi_scattering_lut_height) - 0.5,
            0.0,
            ti.static(self.config.multi_scattering_lut_height - 1),
        )
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(
            x0 + 1,
            ti.static(self.config.multi_scattering_lut_width - 1),
        )
        y1 = ti.min(
            y0 + 1,
            ti.static(self.config.multi_scattering_lut_height - 1),
        )
        fx = x - ti.cast(x0, ti.f32)
        fy = y - ti.cast(y0, ti.f32)
        low = self.multi_scattering_lut[x0, y0] * (
            1.0 - fx
        ) + self.multi_scattering_lut[x1, y0] * fx
        high = self.multi_scattering_lut[x0, y1] * (
            1.0 - fx
        ) + self.multi_scattering_lut[x1, y1] * fx
        return low * (1.0 - fy) + high * fy

    @ti.func
    def _solar_shadow_roots(
        self,
        origin: ti.template(),
        ray: ti.template(),
        sun: ti.template(),
        bottom_radius,
    ):
        """Intersect a view ray with the central solar-shadow cylinder.

        The roots identify where direct-sun transport changes most rapidly.
        They are integration features, not binary shadow decisions: the actual
        finite-disk visibility remains evaluated by `_solar_disk_visibility`.
        """

        origin_perpendicular = origin - sun * origin.dot(sun)
        ray_perpendicular = ray - sun * ray.dot(sun)
        quadratic = ray_perpendicular.dot(ray_perpendicular)
        half_linear = origin_perpendicular.dot(ray_perpendicular)
        perpendicular_length = origin_perpendicular.norm()
        constant = (
            perpendicular_length - bottom_radius
        ) * (
            perpendicular_length + bottom_radius
        )

        root0 = 0.0
        root1 = 0.0
        valid0 = 0
        valid1 = 0
        discriminant = half_linear * half_linear - quadratic * constant
        if quadratic > 1.0e-8 and discriminant >= 0.0:
            square_root = ti.sqrt(ti.max(discriminant, 0.0))
            root0 = (-half_linear - square_root) / quadratic
            root1 = (-half_linear + square_root) / quadratic
            point0 = origin + ray * root0
            point1 = origin + ray * root1
            if point0.dot(sun) < 0.0:
                valid0 = 1
            if point1.dot(sun) < 0.0:
                valid1 = 1
        return root0, root1, valid0, valid1

    @ti.func
    def _solar_disk_visibility(
        self,
        radius,
        sun_cosine,
        bottom_radius,
        angular_radius,
    ):
        """Fraction of a finite solar disk above the spherical horizon."""

        radius = ti.max(radius, bottom_radius)
        horizon_cosine = -ti.sqrt(
            ti.max(
                1.0 - bottom_radius * bottom_radius / (radius * radius),
                0.0,
            )
        )
        horizon_zenith = ti.acos(ti.math.clamp(horizon_cosine, -1.0, 1.0))
        sun_zenith = ti.acos(ti.math.clamp(sun_cosine, -1.0, 1.0))
        signed_separation = horizon_zenith - sun_zenith
        normalized = ti.math.clamp(
            signed_separation / ti.max(angular_radius, 1.0e-6),
            -1.0,
            1.0,
        )
        return (
            ti.acos(-normalized)
            + normalized * ti.sqrt(ti.max(1.0 - normalized * normalized, 0.0))
        ) / _PI

    @ti.func
    def _sample_solar_transmittance(
        self,
        radius,
        sun_cosine,
        bottom_radius,
        angular_radius,
    ):
        visibility = self._solar_disk_visibility(
            radius,
            sun_cosine,
            bottom_radius,
            angular_radius,
        )
        radius = ti.max(radius, bottom_radius)
        horizon_cosine = -ti.sqrt(
            ti.max(
                1.0 - bottom_radius * bottom_radius / (radius * radius),
                0.0,
            )
        )
        horizon_zenith = ti.acos(ti.math.clamp(horizon_cosine, -1.0, 1.0))
        sun_zenith = ti.acos(ti.math.clamp(sun_cosine, -1.0, 1.0))
        # When only part of the disk is visible its centre can lie below the
        # horizon.  Sample optical depth through a representative visible
        # portion instead of feeding a blocked centre ray to the LUT.
        representative_zenith = ti.min(
            sun_zenith,
            horizon_zenith - angular_radius * 0.25,
        )
        representative_cosine = ti.cos(ti.max(representative_zenith, 0.0))
        return visibility * self._sample_transmittance(
            radius,
            representative_cosine,
            bottom_radius,
        )

    @ti.func
    def _rayleigh_phase(self, cosine):
        return 3.0 * (1.0 + cosine * cosine) / (16.0 * _PI)

    @ti.func
    def _mie_phase(self, cosine):
        g = ti.static(self.config.mie_phase_g)
        denominator = ti.max(1.0 + g * g - 2.0 * g * cosine, 1.0e-8)
        return (
            3.0
            * (1.0 - g * g)
            * (1.0 + cosine * cosine)
            / (8.0 * _PI * (2.0 + g * g) * denominator**1.5)
        )

    @ti.func
    def _segment_integral(self, extinction: ti.template(), step):
        transmission = ti.exp(-ti.min(extinction * step, 80.0))
        integral = ti.Vector.zero(ti.f32, 3)
        for channel in ti.static(range(3)):
            integral[channel] = step
            if extinction[channel] > 1.0e-12:
                integral[channel] = (
                    1.0 - transmission[channel]
                ) / extinction[channel]
        return transmission, integral

    @ti.func
    def _scattering_source(
        self,
        point: ti.template(),
        ray: ti.template(),
        sun: ti.template(),
        solar_irradiance: ti.template(),
        bottom_radius,
        sun_angular_radius,
        include_multiple: ti.template(),
    ):
        radius = point.norm()
        radial = point / ti.max(radius, 1.0)
        altitude = radius - bottom_radius
        density = self._density(altitude)
        sun_cosine = radial.dot(sun)
        sun_transmission = self._sample_solar_transmittance(
            radius,
            sun_cosine,
            bottom_radius,
            sun_angular_radius,
        )
        scattering_cosine = ti.math.clamp(ray.dot(sun), -1.0, 1.0)
        source = solar_irradiance * sun_transmission * (
            self._rayleigh_beta()
            * density.x
            * self._rayleigh_phase(scattering_cosine)
            + self._mie_scattering_beta()
            * density.y
            * self._mie_phase(scattering_cosine)
        )
        if ti.static(include_multiple):
            multiple_radiance = self._sample_multi_scattering(
                radius,
                sun_cosine,
                bottom_radius,
            )
            source += self._scattering(altitude) * multiple_radiance
        return source

    @ti.func
    def _scattering_segment(
        self,
        origin: ti.template(),
        ray: ti.template(),
        start,
        end,
        sun: ti.template(),
        solar_irradiance: ti.template(),
        bottom_radius,
        sun_angular_radius,
    ):
        """Integrate one constant-source segment and return `(L, T)`."""

        step = ti.max(end - start, 0.0)
        distance = (start + end) * 0.5
        point = origin + ray * distance
        altitude = point.norm() - bottom_radius
        extinction = self._extinction(altitude)
        transmission, extinction_integral = self._segment_integral(
            extinction,
            step,
        )
        source = self._scattering_source(
            point,
            ray,
            sun,
            solar_irradiance,
            bottom_radius,
            sun_angular_radius,
            True,
        )
        return source * extinction_integral, transmission

    @ti.func
    def _terminator_refinement_weight(
        self,
        origin: ti.template(),
        ray: ti.template(),
        start,
        end,
        sun: ti.template(),
        bottom_radius,
        sun_angular_radius,
        shadow_root0,
        shadow_root1,
        shadow_root0_valid,
        shadow_root1_valid,
    ):
        """Continuous importance for intervals near the solar terminator.

        Root proximity catches a narrow transition even when no quadrature
        sample happens to land in the penumbra.  Midpoint disk visibility also
        handles rays nearly parallel to the shadow cylinder, where finite-disk
        penumbra exists without a well-conditioned cylinder crossing.
        """

        midpoint = (start + end) * 0.5
        interval_width = ti.max(end - start, 1.0e-4)
        closest_root = 1.0e30
        if shadow_root0_valid != 0:
            closest_root = ti.min(
                closest_root,
                ti.abs(midpoint - shadow_root0),
            )
        if shadow_root1_valid != 0:
            closest_root = ti.min(
                closest_root,
                ti.abs(midpoint - shadow_root1),
            )

        # Fade refinement across two neighboring base intervals so changing
        # camera rays cannot expose a hard adaptive-sampling boundary.
        proximity = ti.math.clamp(
            2.0 - closest_root / interval_width,
            0.0,
            1.0,
        )
        proximity = proximity * proximity * (3.0 - 2.0 * proximity)

        point = origin + ray * midpoint
        radius = point.norm()
        radial = point / ti.max(radius, 1.0)
        visibility = self._solar_disk_visibility(
            radius,
            radial.dot(sun),
            bottom_radius,
            sun_angular_radius,
        )
        penumbra = 4.0 * visibility * (1.0 - visibility)
        return ti.max(proximity, penumbra)

    @ti.func
    def _surface_terminator_weight(
        self,
        origin: ti.template(),
        ray: ti.template(),
        surface_distance,
        sun: ti.template(),
        bottom_radius,
        sun_angular_radius,
    ):
        """Select direct integration for surface paths crossing a terminator."""

        weight = 0.0
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        outer_near, outer_far, outer_hit = self._sphere_roots(
            origin,
            ray,
            top_radius,
        )
        start = ti.max(outer_near, 0.0)
        end = ti.min(surface_distance, outer_far)
        if outer_hit and end > start:
            root0, root1, valid0, valid1 = self._solar_shadow_roots(
                origin,
                ray,
                sun,
                bottom_radius,
            )
            # The finite disk turns the ideal cylinder into a penumbra whose
            # characteristic width grows with planet radius and angular size.
            fade_distance = ti.max(
                bottom_radius * sun_angular_radius * 2.0,
                1.0,
            )
            closest = 1.0e30
            if valid0 != 0:
                root_distance = ti.max(
                    ti.max(start - root0, root0 - end),
                    0.0,
                )
                closest = ti.min(closest, root_distance)
            if valid1 != 0:
                root_distance = ti.max(
                    ti.max(start - root1, root1 - end),
                    0.0,
                )
                closest = ti.min(closest, root_distance)
            weight = ti.math.clamp(
                1.0 - closest / fade_distance,
                0.0,
                1.0,
            )
            weight = weight * weight * (3.0 - 2.0 * weight)
        return weight

    @ti.func
    def _integrate_adaptive_scattering_interval(
        self,
        origin: ti.template(),
        ray: ti.template(),
        start,
        end,
        sun: ti.template(),
        solar_irradiance: ti.template(),
        bottom_radius,
        sun_angular_radius,
        refinement_weight,
    ):
        """Blend base and terminator-refined quadrature continuously."""

        coarse_radiance, coarse_transmission = self._scattering_segment(
            origin,
            ray,
            start,
            end,
            sun,
            solar_irradiance,
            bottom_radius,
            sun_angular_radius,
        )
        result_radiance = coarse_radiance
        result_transmission = coarse_transmission

        if refinement_weight > 0.0:
            fine_radiance = ti.Vector.zero(ti.f32, 3)
            fine_transmission = ti.Vector([1.0, 1.0, 1.0])
            substep = 0
            while substep < ti.static(self.config.aerial_terminator_substeps):
                fraction0 = ti.cast(substep, ti.f32) / ti.static(
                    self.config.aerial_terminator_substeps
                )
                fraction1 = ti.cast(substep + 1, ti.f32) / ti.static(
                    self.config.aerial_terminator_substeps
                )
                child_start = start + (end - start) * fraction0
                child_end = start + (end - start) * fraction1
                child_radiance, child_transmission = self._scattering_segment(
                    origin,
                    ray,
                    child_start,
                    child_end,
                    sun,
                    solar_irradiance,
                    bottom_radius,
                    sun_angular_radius,
                )
                fine_radiance += fine_transmission * child_radiance
                fine_transmission *= child_transmission
                substep += 1

            result_radiance = coarse_radiance * (
                1.0 - refinement_weight
            ) + fine_radiance * refinement_weight
            result_transmission = coarse_transmission * (
                1.0 - refinement_weight
            ) + fine_transmission * refinement_weight
        return result_radiance, result_transmission

    @ti.kernel
    def _build_transmittance(self, bottom_radius: ti.f32):
        for x, y in self.transmittance_lut:
            u = (ti.cast(x, ti.f32) + 0.5) / ti.static(
                self.config.transmittance_lut_width
            )
            v = (ti.cast(y, ti.f32) + 0.5) / ti.static(
                self.config.transmittance_lut_height
            )
            radius, cosine, distance = self._transmittance_uv_to_ray(
                u,
                v,
                bottom_radius,
            )
            sine = ti.sqrt(ti.max(1.0 - cosine * cosine, 0.0))
            origin = ti.Vector([0.0, radius, 0.0])
            ray = ti.Vector([sine, cosine, 0.0])
            transmission = self._integrate_extinction(
                origin,
                ray,
                0.0,
                distance,
                bottom_radius,
            )
            self.transmittance_lut[x, y] = ti.math.clamp(
                transmission,
                0.0,
                1.0,
            )

    @ti.kernel
    def _build_multi_scattering(
        self,
        bottom_radius: ti.f32,
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_angular_radius: ti.f32,
    ):
        direction_count = ti.static(self.config.multi_scattering_directions)
        path_steps = ti.static(self.config.multi_scattering_steps)
        for x, y in self.multi_scattering_lut:
            u = (ti.cast(x, ti.f32) + 0.5) / ti.static(
                self.config.multi_scattering_lut_width
            )
            v = (ti.cast(y, ti.f32) + 0.5) / ti.static(
                self.config.multi_scattering_lut_height
            )
            radius = bottom_radius + v * v * ti.static(
                self.config.top_altitude_m
            )
            sun_cosine = self._sun_u_to_cosine(
                u,
                radius,
                bottom_radius,
            )
            sun_sine = ti.sqrt(ti.max(1.0 - sun_cosine * sun_cosine, 0.0))
            sun = ti.Vector([sun_sine, sun_cosine, 0.0])
            origin = ti.Vector([0.0, radius, 0.0])
            luminance_sum = ti.Vector.zero(ti.f32, 3)
            feedback_sum = ti.Vector.zero(ti.f32, 3)

            direction_index = 0
            while direction_index < direction_count:
                sample_index = ti.cast(direction_index, ti.f32) + 0.5
                direction_cosine = 1.0 - 2.0 * sample_index / direction_count
                direction_sine = ti.sqrt(
                    ti.max(1.0 - direction_cosine * direction_cosine, 0.0)
                )
                azimuth = ti.cast(direction_index, ti.f32) * 2.39996323
                ray = ti.Vector(
                    [
                        direction_sine * ti.cos(azimuth),
                        direction_cosine,
                        direction_sine * ti.sin(azimuth),
                    ]
                )
                start, end, valid, ground = self._atmosphere_segment(
                    origin,
                    ray,
                    bottom_radius,
                )
                direction_radiance = ti.Vector.zero(ti.f32, 3)
                direction_feedback = ti.Vector.zero(ti.f32, 3)
                path_transmission = ti.Vector([1.0, 1.0, 1.0])
                if valid:
                    path_index = 0
                    while path_index < path_steps:
                        fraction0 = ti.cast(path_index, ti.f32) / path_steps
                        fraction1 = ti.cast(path_index + 1, ti.f32) / path_steps
                        distance0 = self._warped_interval_boundary(
                            origin,
                            ray,
                            start,
                            end,
                            fraction0,
                        )
                        distance1 = self._warped_interval_boundary(
                            origin,
                            ray,
                            start,
                            end,
                            fraction1,
                        )
                        distance = (distance0 + distance1) * 0.5
                        step = distance1 - distance0
                        point = origin + ray * distance
                        altitude = point.norm() - bottom_radius
                        extinction = self._extinction(altitude)
                        segment_transmission, segment_integral = (
                            self._segment_integral(extinction, step)
                        )
                        direct_source = self._scattering_source(
                            point,
                            ray,
                            sun,
                            solar_irradiance,
                            bottom_radius,
                            sun_angular_radius,
                            False,
                        )
                        direction_radiance += (
                            path_transmission
                            * direct_source
                            * segment_integral
                        )
                        direction_feedback += (
                            path_transmission
                            * self._scattering(altitude)
                            * segment_integral
                        )
                        path_transmission *= segment_transmission
                        path_index += 1

                    if ground:
                        ground_point = origin + ray * end
                        ground_normal = ground_point.normalized()
                        ground_sun_cosine = ground_normal.dot(sun)
                        ground_transmission = self._sample_solar_transmittance(
                            bottom_radius,
                            ground_sun_cosine,
                            bottom_radius,
                            sun_angular_radius,
                        )
                        ground_radiance = (
                            self._ground_albedo()
                            * solar_irradiance
                            * ground_transmission
                            * ti.max(ground_sun_cosine, 0.0)
                            / _PI
                        )
                        direction_radiance += path_transmission * ground_radiance

                luminance_sum += direction_radiance
                feedback_sum += direction_feedback
                direction_index += 1

            average_luminance = luminance_sum / direction_count
            average_feedback = ti.math.clamp(
                feedback_sum / direction_count,
                0.0,
                0.95,
            )
            self.multi_scattering_lut[x, y] = ti.max(
                average_luminance / (1.0 - average_feedback),
                0.0,
            )

    @ti.kernel
    def _build_sky_view(
        self,
        bottom_radius: ti.f32,
        camera_radius: ti.f32,
        sun_zenith_cosine: ti.f32,
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_angular_radius: ti.f32,
    ):
        sun_sine = ti.sqrt(ti.max(1.0 - sun_zenith_cosine**2, 0.0))
        sun = ti.Vector([sun_sine, sun_zenith_cosine, 0.0])
        safe_camera_radius = ti.max(camera_radius, bottom_radius + 1.0e-3)
        origin = ti.Vector([0.0, safe_camera_radius, 0.0])
        for x, y in self.sky_view_lut:
            azimuth = (
                (ti.cast(x, ti.f32) + 0.5)
                / ti.static(self.config.sky_view_lut_width)
                * _PI
            )
            view_v = (ti.cast(y, ti.f32) + 0.5) / ti.static(
                self.config.sky_view_lut_height
            )
            zenith = self._sky_v_to_zenith(
                view_v,
                safe_camera_radius,
                bottom_radius,
            )
            view_cosine = ti.cos(zenith)
            view_sine = ti.sin(zenith)
            ray = ti.Vector(
                [
                    view_sine * ti.cos(azimuth),
                    view_cosine,
                    view_sine * ti.sin(azimuth),
                ]
            )
            start, end, valid, _ = self._atmosphere_segment(
                origin,
                ray,
                bottom_radius,
            )
            radiance = ti.Vector.zero(ti.f32, 3)
            if valid:
                view_transmission = ti.Vector([1.0, 1.0, 1.0])
                index = 0
                while index < ti.static(self.config.sky_view_steps):
                    fraction0 = ti.cast(index, ti.f32) / ti.static(
                        self.config.sky_view_steps
                    )
                    fraction1 = ti.cast(index + 1, ti.f32) / ti.static(
                        self.config.sky_view_steps
                    )
                    distance0 = self._warped_interval_boundary(
                        origin,
                        ray,
                        start,
                        end,
                        fraction0,
                    )
                    distance1 = self._warped_interval_boundary(
                        origin,
                        ray,
                        start,
                        end,
                        fraction1,
                    )
                    distance = (distance0 + distance1) * 0.5
                    step = distance1 - distance0
                    point = origin + ray * distance
                    altitude = point.norm() - bottom_radius
                    source = self._scattering_source(
                        point,
                        ray,
                        sun,
                        solar_irradiance,
                        bottom_radius,
                        sun_angular_radius,
                        True,
                    )
                    extinction = self._extinction(altitude)
                    segment_transmission, segment_integral = self._segment_integral(
                        extinction,
                        step,
                    )
                    radiance += view_transmission * source * segment_integral
                    view_transmission *= segment_transmission
                    index += 1
            self.sky_view_lut[x, y] = ti.max(radiance, 0.0)

    @ti.kernel
    def _build_aerial_perspective(
        self,
        bottom_radius: ti.f32,
        camera_radius: ti.f32,
        sun: ti.types.vector(3, ti.f32),
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_angular_radius: ti.f32,
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
        tangent_half_fov: ti.f32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        origin = ti.Vector([0.0, camera_radius, 0.0])
        for x, y in ti.ndrange(
            ti.static(self.config.aerial_lut_width),
            ti.static(self.config.aerial_lut_height),
        ):
            u = (ti.cast(x, ti.f32) + 0.5) / ti.static(
                self.config.aerial_lut_width
            )
            v = (ti.cast(y, ti.f32) + 0.5) / ti.static(
                self.config.aerial_lut_height
            )
            sx = (u * 2.0 - 1.0) * aspect
            sy = v * 2.0 - 1.0
            ray = (
                right * sx + view_up * sy + forward / tangent_half_fov
            ).normalized()
            atmosphere_start, atmosphere_end, valid = self._aerial_prefix_segment(
                origin,
                ray,
                bottom_radius,
            )
            cumulative_radiance = ti.Vector.zero(ti.f32, 3)
            cumulative_transmission = ti.Vector([1.0, 1.0, 1.0])
            previous_target = atmosphere_start
            depth_index = 0
            while depth_index < ti.static(self.config.aerial_lut_depth):
                depth_fraction = ti.cast(depth_index, ti.f32) / ti.static(
                    self.config.aerial_lut_depth - 1
                )
                # Parameterize depth over this ray's actual participating
                # medium interval. A camera-distance volume wastes almost all
                # slices before atmosphere entry for an orbital camera and
                # quantizes the final 100 km shell into a handful of bands.
                target = atmosphere_start + (
                    atmosphere_end - atmosphere_start
                ) * depth_fraction
                if valid:
                    interval_start = ti.max(previous_target, atmosphere_start)
                    interval_end = ti.min(target, atmosphere_end)
                    if interval_end > interval_start:
                        sample_index = 0
                        while sample_index < ti.static(
                            self.config.aerial_steps_per_slice
                        ):
                            fraction0 = ti.cast(sample_index, ti.f32) / ti.static(
                                self.config.aerial_steps_per_slice
                            )
                            fraction1 = ti.cast(
                                sample_index + 1,
                                ti.f32,
                            ) / ti.static(self.config.aerial_steps_per_slice)
                            distance0 = self._warped_interval_boundary(
                                origin,
                                ray,
                                interval_start,
                                interval_end,
                                fraction0,
                            )
                            distance1 = self._warped_interval_boundary(
                                origin,
                                ray,
                                interval_start,
                                interval_end,
                                fraction1,
                            )
                            distance = (distance0 + distance1) * 0.5
                            step = distance1 - distance0
                            point = origin + ray * distance
                            altitude = point.norm() - bottom_radius
                            extinction = self._extinction(altitude)
                            segment_transmission, segment_integral = (
                                self._segment_integral(extinction, step)
                            )
                            source = self._scattering_source(
                                point,
                                ray,
                                sun,
                                solar_irradiance,
                                bottom_radius,
                                sun_angular_radius,
                                True,
                            )
                            cumulative_radiance += (
                                cumulative_transmission
                                * source
                                * segment_integral
                            )
                            cumulative_transmission *= segment_transmission
                            sample_index += 1
                self.aerial_scattering_lut[x, y, depth_index] = ti.max(
                    cumulative_radiance,
                    0.0,
                )
                previous_target = target
                depth_index += 1

    @ti.kernel
    def _prepare_surface_lighting(
        self,
        position_view: ti.template(),
        surface_id: ti.template(),
        bottom_radius: ti.f32,
        camera_radius: ti.f32,
        sun: ti.types.vector(3, ti.f32),
        sun_angular_radius: ti.f32,
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
    ):
        camera_planet_position = ti.Vector([0.0, camera_radius, 0.0])
        for pixel in ti.grouped(surface_id):
            solar_transmission = ti.Vector([1.0, 1.0, 1.0])
            sky_radiance = ti.Vector.zero(ti.f32, 3)
            if surface_id[pixel] >= 0:
                view_position = position_view[pixel]
                local_position = (
                    right * view_position.x
                    + view_up * view_position.y
                    + forward * view_position.z
                )
                planet_position = camera_planet_position + local_position
                radius = planet_position.norm()
                radial = planet_position / ti.max(radius, 1.0)
                sun_cosine = radial.dot(sun)
                solar_transmission = self._sample_solar_transmittance(
                    radius,
                    sun_cosine,
                    bottom_radius,
                    sun_angular_radius,
                )
                sky_radiance = self._sample_multi_scattering(
                    radius,
                    sun_cosine,
                    bottom_radius,
                )
            self.surface_sun_transmittance[pixel] = solar_transmission
            self.surface_sky_radiance[pixel] = ti.max(sky_radiance, 0.0)

    @ti.func
    def _sample_sky_view(
        self,
        ray: ti.template(),
        sun: ti.template(),
        camera_radius,
        bottom_radius,
    ):
        zenith = ti.acos(ti.math.clamp(ray.y, -1.0, 1.0))
        view_horizontal = ti.Vector([ray.x, ray.z])
        sun_horizontal = ti.Vector([sun.x, sun.z])
        view_length = view_horizontal.norm()
        sun_length = sun_horizontal.norm()
        relative_azimuth = 0.0
        if view_length > 1.0e-6 and sun_length > 1.0e-6:
            relative_azimuth = ti.acos(
                ti.math.clamp(
                    view_horizontal.dot(sun_horizontal)
                    / (view_length * sun_length),
                    -1.0,
                    1.0,
                )
            )
        u = relative_azimuth / _PI
        v = self._sky_zenith_to_v(
            zenith,
            camera_radius,
            bottom_radius,
        )
        x = ti.math.clamp(
            u * ti.static(self.config.sky_view_lut_width) - 0.5,
            0.0,
            ti.static(self.config.sky_view_lut_width - 1),
        )
        y = ti.math.clamp(
            v * ti.static(self.config.sky_view_lut_height) - 0.5,
            0.0,
            ti.static(self.config.sky_view_lut_height - 1),
        )
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(x0 + 1, ti.static(self.config.sky_view_lut_width - 1))
        y1 = ti.min(y0 + 1, ti.static(self.config.sky_view_lut_height - 1))
        fx = x - ti.cast(x0, ti.f32)
        fy = y - ti.cast(y0, ti.f32)
        low = self.sky_view_lut[x0, y0] * (1.0 - fx) + self.sky_view_lut[
            x1, y0
        ] * fx
        high = self.sky_view_lut[x0, y1] * (1.0 - fx) + self.sky_view_lut[
            x1, y1
        ] * fx
        return low * (1.0 - fy) + high * fy

    @ti.func
    def _sample_aerial_field(
        self,
        field: ti.template(),
        screen_u,
        screen_v,
        depth_fraction,
    ):
        x = ti.math.clamp(
            screen_u * ti.static(self.config.aerial_lut_width) - 0.5,
            0.0,
            ti.static(self.config.aerial_lut_width - 1),
        )
        y = ti.math.clamp(
            screen_v * ti.static(self.config.aerial_lut_height) - 0.5,
            0.0,
            ti.static(self.config.aerial_lut_height - 1),
        )
        z = ti.math.clamp(depth_fraction, 0.0, 1.0) * ti.static(
            self.config.aerial_lut_depth - 1
        )
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        z0 = ti.cast(ti.floor(z), ti.i32)
        x1 = ti.min(x0 + 1, ti.static(self.config.aerial_lut_width - 1))
        y1 = ti.min(y0 + 1, ti.static(self.config.aerial_lut_height - 1))
        z1 = ti.min(z0 + 1, ti.static(self.config.aerial_lut_depth - 1))
        fx = x - ti.cast(x0, ti.f32)
        fy = y - ti.cast(y0, ti.f32)
        fz = z - ti.cast(z0, ti.f32)
        z_low_y_low = field[x0, y0, z0] * (1.0 - fx) + field[
            x1, y0, z0
        ] * fx
        z_low_y_high = field[x0, y1, z0] * (1.0 - fx) + field[
            x1, y1, z0
        ] * fx
        z_high_y_low = field[x0, y0, z1] * (1.0 - fx) + field[
            x1, y0, z1
        ] * fx
        z_high_y_high = field[x0, y1, z1] * (1.0 - fx) + field[
            x1, y1, z1
        ] * fx
        low = z_low_y_low * (1.0 - fy) + z_low_y_high * fy
        high = z_high_y_low * (1.0 - fy) + z_high_y_high * fy
        return low * (1.0 - fz) + high * fz

    @ti.func
    def _sample_aerial_scattering(
        self,
        screen_u,
        screen_v,
        distance,
        camera_radius,
        ray: ti.template(),
        bottom_radius,
    ):
        origin = ti.Vector([0.0, camera_radius, 0.0])
        start, end, valid = self._aerial_prefix_segment(
            origin,
            ray,
            bottom_radius,
        )
        depth_fraction = 0.0
        if valid:
            depth_fraction = ti.math.clamp(
                (distance - start) / ti.max(end - start, 1.0),
                0.0,
                1.0,
            )
        scattering = self._sample_aerial_field(
            self.aerial_scattering_lut,
            screen_u,
            screen_v,
            depth_fraction,
        )
        return scattering

    @ti.func
    def _surface_segment_transmittance(
        self,
        origin: ti.template(),
        ray: ti.template(),
        surface_distance,
        bottom_radius,
    ):
        """Reconstruct exact camera-to-surface T from two 2-D LUT samples.

        Sampling from the surface back toward the camera makes both lookups
        unoccluded.  Their ratio isolates the actual atmospheric segment and
        avoids angular interpolation through a low-resolution froxel volume.
        """

        transmission = ti.Vector([1.0, 1.0, 1.0])
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        outer_near, outer_far, outer_hit = self._sphere_roots(
            origin,
            ray,
            top_radius,
        )
        start = ti.max(outer_near, 0.0)
        end = ti.min(surface_distance, outer_far)
        if outer_hit and end > start:
            near_point = origin + ray * start
            surface_point = origin + ray * end
            reverse_ray = -ray

            near_length = near_point.norm()
            surface_length = surface_point.norm()
            near_radius = ti.max(near_length, bottom_radius)
            surface_radius = ti.max(surface_length, bottom_radius)
            near_radial = near_point / ti.max(near_length, 1.0)
            surface_radial = surface_point / ti.max(surface_length, 1.0)

            surface_to_top = self._sample_transmittance_unoccluded(
                surface_radius,
                surface_radial.dot(reverse_ray),
                bottom_radius,
            )
            near_to_top = self._sample_transmittance_unoccluded(
                near_radius,
                near_radial.dot(reverse_ray),
                bottom_radius,
            )
            transmission = ti.math.clamp(
                surface_to_top / ti.max(near_to_top, 1.0e-6),
                0.0,
                1.0,
            )
        return transmission

    @ti.func
    def _integrate_surface_scattering(
        self,
        origin: ti.template(),
        ray: ti.template(),
        surface_distance,
        sun: ti.template(),
        solar_irradiance: ti.template(),
        bottom_radius,
        sun_angular_radius,
    ):
        """Integrate the true camera-to-G-buffer path for limb pixels."""

        radiance = ti.Vector.zero(ti.f32, 3)
        view_transmission = ti.Vector([1.0, 1.0, 1.0])
        top_radius = bottom_radius + ti.static(self.config.top_altitude_m)
        outer_near, outer_far, outer_hit = self._sphere_roots(
            origin,
            ray,
            top_radius,
        )
        start = ti.max(outer_near, 0.0)
        end = ti.min(surface_distance, outer_far)
        if outer_hit and end > start:
            shadow_root0, shadow_root1, shadow_valid0, shadow_valid1 = (
                self._solar_shadow_roots(
                    origin,
                    ray,
                    sun,
                    bottom_radius,
                )
            )
            sample_index = 0
            while sample_index < ti.static(
                self.config.aerial_horizon_raymarch_steps
            ):
                fraction0 = ti.cast(sample_index, ti.f32) / ti.static(
                    self.config.aerial_horizon_raymarch_steps
                )
                fraction1 = ti.cast(sample_index + 1, ti.f32) / ti.static(
                    self.config.aerial_horizon_raymarch_steps
                )
                distance0 = self._warped_interval_boundary(
                    origin,
                    ray,
                    start,
                    end,
                    fraction0,
                )
                distance1 = self._warped_interval_boundary(
                    origin,
                    ray,
                    start,
                    end,
                    fraction1,
                )
                refinement_weight = self._terminator_refinement_weight(
                    origin,
                    ray,
                    distance0,
                    distance1,
                    sun,
                    bottom_radius,
                    sun_angular_radius,
                    shadow_root0,
                    shadow_root1,
                    shadow_valid0,
                    shadow_valid1,
                )
                interval_radiance, interval_transmission = (
                    self._integrate_adaptive_scattering_interval(
                        origin,
                        ray,
                        distance0,
                        distance1,
                        sun,
                        solar_irradiance,
                        bottom_radius,
                        sun_angular_radius,
                        refinement_weight,
                    )
                )
                radiance += view_transmission * interval_radiance
                view_transmission *= interval_transmission
                sample_index += 1
        return ti.max(radiance, 0.0)

    @ti.func
    def _surface_aerial_scattering(
        self,
        screen_u,
        screen_v,
        surface_distance,
        camera_radius,
        ray: ti.template(),
        sun: ti.template(),
        solar_irradiance: ti.template(),
        bottom_radius,
        sun_angular_radius,
    ):
        """Use froxels in smooth regions and true integration at the limb."""

        froxel = self._sample_aerial_scattering(
            screen_u,
            screen_v,
            surface_distance,
            camera_radius,
            ray,
            bottom_radius,
        )
        origin = ti.Vector([0.0, camera_radius, 0.0])
        surface_point = origin + ray * surface_distance
        surface_radius = surface_point.norm()
        surface_radial = surface_point / ti.max(surface_radius, 1.0)
        limb_cosine = ti.max(surface_radial.dot(-ray), 0.0)
        inner = ti.static(self.config.aerial_horizon_inner_cosine)
        outer = ti.static(self.config.aerial_horizon_outer_cosine)
        direct_weight = ti.math.clamp(
            (outer - limb_cosine) / ti.max(outer - inner, 1.0e-6),
            0.0,
            1.0,
        )
        direct_weight = direct_weight * direct_weight * (
            3.0 - 2.0 * direct_weight
        )
        terminator_weight = self._surface_terminator_weight(
            origin,
            ray,
            surface_distance,
            sun,
            bottom_radius,
            sun_angular_radius,
        )
        direct_weight = ti.max(direct_weight, terminator_weight)
        result = froxel
        if direct_weight > 0.0:
            direct = self._integrate_surface_scattering(
                origin,
                ray,
                surface_distance,
                sun,
                solar_irradiance,
                bottom_radius,
                sun_angular_radius,
            )
            result = froxel * (1.0 - direct_weight) + direct * direct_weight
        return result

    @ti.func
    def _camera_transmittance(
        self,
        camera_radius,
        ray: ti.template(),
        bottom_radius,
    ):
        origin = ti.Vector([0.0, camera_radius, 0.0])
        start, _, valid, ground = self._atmosphere_segment(
            origin,
            ray,
            bottom_radius,
        )
        transmission = ti.Vector([1.0, 1.0, 1.0])
        if valid:
            if ground:
                transmission = ti.Vector.zero(ti.f32, 3)
            else:
                entry = origin + ray * start
                radius = entry.norm()
                radial = entry / ti.max(radius, 1.0)
                transmission = self._sample_transmittance(
                    radius,
                    radial.dot(ray),
                    bottom_radius,
                )
        return transmission

    @ti.kernel
    def _composite(
        self,
        scene_hdr: ti.template(),
        surface_id: ti.template(),
        position_view: ti.template(),
        hdr: ti.template(),
        bottom_radius: ti.f32,
        camera_radius: ti.f32,
        sky_lut_camera_radius: ti.f32,
        sun_local: ti.types.vector(3, ti.f32),
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_disk_radiance: ti.types.vector(3, ti.f32),
        sun_angular_radius: ti.f32,
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
        tangent_half_fov: ti.f32,
        diagnostic_view: ti.i32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        for pixel in ti.grouped(hdr):
            color = scene_hdr[pixel]
            screen_u = (ti.cast(pixel.x, ti.f32) + 0.5) / self.width
            screen_v = (ti.cast(pixel.y, ti.f32) + 0.5) / self.height
            sx = (screen_u * 2.0 - 1.0) * aspect
            sy = screen_v * 2.0 - 1.0
            ray = (
                right * sx + view_up * sy + forward / tangent_half_fov
            ).normalized()
            if diagnostic_view == _DIAGNOSTIC_COMPOSITE:
                if surface_id[pixel] >= 0:
                    surface_distance = position_view[pixel].norm()
                    scattering = self._surface_aerial_scattering(
                        screen_u,
                        screen_v,
                        surface_distance,
                        camera_radius,
                        ray,
                        sun_local,
                        solar_irradiance,
                        bottom_radius,
                        sun_angular_radius,
                    )
                    transmission = self._surface_segment_transmittance(
                        ti.Vector([0.0, camera_radius, 0.0]),
                        ray,
                        surface_distance,
                        bottom_radius,
                    )
                    color = scattering + color * transmission
                else:
                    transmission = self._camera_transmittance(
                        camera_radius,
                        ray,
                        bottom_radius,
                    )
                    color = (
                        self._sample_sky_view(
                            ray,
                            sun_local,
                            sky_lut_camera_radius,
                            bottom_radius,
                        )
                        + color * transmission
                    )
                    angular_distance = ti.acos(
                        ti.math.clamp(ray.dot(sun_local), -1.0, 1.0)
                    )
                    # One-pixel radial filter. The previous binary centre
                    # sample made a small finite disk stair-step and allowed
                    # Bloom to define its apparent (square) silhouette.
                    pixel_filter = ti.atan2(
                        tangent_half_fov * 1.41421356 / ti.cast(self.height, ti.f32),
                        1.0,
                    )
                    coverage = ti.math.clamp(
                        (
                            sun_angular_radius
                            + pixel_filter
                            - angular_distance
                        )
                        / ti.max(2.0 * pixel_filter, 1.0e-8),
                        0.0,
                        1.0,
                    )
                    coverage = coverage * coverage * (3.0 - 2.0 * coverage)
                    color += sun_disk_radiance * transmission * coverage
            elif diagnostic_view == _DIAGNOSTIC_SKY_VIEW:
                color = self._sample_sky_view(
                    ray,
                    sun_local,
                    sky_lut_camera_radius,
                    bottom_radius,
                )
            elif diagnostic_view == _DIAGNOSTIC_CAMERA_TRANSMITTANCE:
                color = self._camera_transmittance(
                    camera_radius,
                    ray,
                    bottom_radius,
                )
            elif diagnostic_view == _DIAGNOSTIC_TRANSMITTANCE_LUT:
                lut_x = ti.min(
                    ti.cast(
                        screen_u
                        * ti.static(self.config.transmittance_lut_width),
                        ti.i32,
                    ),
                    ti.static(self.config.transmittance_lut_width - 1),
                )
                lut_y = ti.min(
                    ti.cast(
                        screen_v
                        * ti.static(self.config.transmittance_lut_height),
                        ti.i32,
                    ),
                    ti.static(self.config.transmittance_lut_height - 1),
                )
                color = self.transmittance_lut[lut_x, lut_y]
            elif diagnostic_view == _DIAGNOSTIC_MULTI_SCATTERING_LUT:
                lut_x = ti.min(
                    ti.cast(
                        screen_u
                        * ti.static(self.config.multi_scattering_lut_width),
                        ti.i32,
                    ),
                    ti.static(self.config.multi_scattering_lut_width - 1),
                )
                lut_y = ti.min(
                    ti.cast(
                        screen_v
                        * ti.static(self.config.multi_scattering_lut_height),
                        ti.i32,
                    ),
                    ti.static(self.config.multi_scattering_lut_height - 1),
                )
                color = self.multi_scattering_lut[lut_x, lut_y]
            elif diagnostic_view == _DIAGNOSTIC_AERIAL_SCATTERING:
                color = ti.Vector.zero(ti.f32, 3)
                if surface_id[pixel] >= 0:
                    color = self._surface_aerial_scattering(
                        screen_u,
                        screen_v,
                        position_view[pixel].norm(),
                        camera_radius,
                        ray,
                        sun_local,
                        solar_irradiance,
                        bottom_radius,
                        sun_angular_radius,
                    )
            elif diagnostic_view == _DIAGNOSTIC_AERIAL_TRANSMITTANCE:
                color = ti.Vector.zero(ti.f32, 3)
                if surface_id[pixel] >= 0:
                    color = self._surface_segment_transmittance(
                        ti.Vector([0.0, camera_radius, 0.0]),
                        ray,
                        position_view[pixel].norm(),
                        bottom_radius,
                    )
            else:
                mask = 0.0
                if surface_id[pixel] >= 0:
                    mask = 1.0
                color = ti.Vector([mask, mask, mask])

            color = ti.max(color, 0.0)
            hdr[pixel] = color

    def set_luts_frozen(self, frozen: bool) -> None:
        """Freeze or resume view-dependent atmosphere LUT regeneration.

        Static Transmittance and Multi-Scattering data remain valid.  Sky-view
        and Aerial-Perspective retain the exact camera state captured before
        the freeze, which makes discontinuities distinguishable from terrain
        or camera motion.  Unfreezing invalidates Sky-View explicitly so no
        movement performed during the freeze can leave stale data.
        """

        frozen = bool(frozen)
        if self.luts_frozen and not frozen:
            self._sky_key = None
        self.luts_frozen = frozen

    def update(
        self,
        planet_radius_m: float,
        camera_radius_m: float,
        sun_local: np.ndarray,
        solar_irradiance: tuple[float, float, float],
        sun_angular_radius_degrees: float,
    ) -> None:
        """Regenerate only LUTs invalidated by atmosphere, altitude or sun state."""

        planet_radius = float(planet_radius_m)
        camera_radius = float(camera_radius_m)
        sun = np.asarray(sun_local, dtype=np.float64)
        sun /= max(float(np.linalg.norm(sun)), 1.0e-12)
        sun_angular_radius = float(
            np.float32(math.radians(sun_angular_radius_degrees))
        )
        if self._planet_radius_key != planet_radius:
            self._build_transmittance(planet_radius)
            self._planet_radius_key = planet_radius
            self._multi_scattering_key = None
            self._sky_key = None
            self.transmittance_rebuilds += 1

        irradiance_key = tuple(float(value) for value in solar_irradiance)
        multi_scattering_key = (
            planet_radius,
            irradiance_key,
            sun_angular_radius,
        )
        if multi_scattering_key != self._multi_scattering_key:
            self._build_multi_scattering(
                planet_radius,
                solar_irradiance,
                sun_angular_radius,
            )
            self._multi_scattering_key = multi_scattering_key
            self._sky_key = None
            self.multi_scattering_rebuilds += 1

        # Key the view LUT by the exact values representable by its f32
        # kernels.  The former altitude and sun-angle buckets made the whole
        # sky remain stale and then jump at a bucket boundary.
        camera_radius_key = float(np.float32(camera_radius))
        sun_zenith_key = float(np.float32(np.clip(sun[1], -1.0, 1.0)))
        sky_key = (
            camera_radius_key,
            sun_zenith_key,
            irradiance_key + (sun_angular_radius,),
        )
        if sky_key != self._sky_key and (
            not self.luts_frozen or self._sky_key is None
        ):
            self._build_sky_view(
                planet_radius,
                camera_radius_key,
                sun_zenith_key,
                solar_irradiance,
                sun_angular_radius,
            )
            self._sky_key = sky_key
            self.sky_snapshot_altitude_m = camera_radius_key - planet_radius
            self.sky_snapshot_sun_cosine = sun_zenith_key
            self.sky_view_rebuilds += 1

    def prepare_frame(
        self,
        position_view,
        surface_id,
        planet_radius_m: float,
        camera_radius_m: float,
        sun_local: np.ndarray,
        solar_irradiance: tuple[float, float, float],
        sun_angular_radius_degrees: float,
        view_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
        tangent_half_fov: float,
    ) -> None:
        """Build view-dependent aerial data and surface-lighting inputs."""

        right, view_up, forward = view_basis
        camera_radius = float(camera_radius_m)
        planet_radius = float(planet_radius_m)
        sun = tuple(float(value) for value in sun_local)
        sun_angular_radius = math.radians(sun_angular_radius_degrees)
        basis_right = tuple(float(value) for value in right)
        basis_up = tuple(float(value) for value in view_up)
        basis_forward = tuple(float(value) for value in forward)
        self._prepare_surface_lighting(
            position_view,
            surface_id,
            float(planet_radius_m),
            float(camera_radius_m),
            sun,
            sun_angular_radius,
            basis_right,
            basis_up,
            basis_forward,
        )
        if not self.luts_frozen or not self._aerial_available:
            self._build_aerial_perspective(
                float(planet_radius_m),
                float(camera_radius_m),
                sun,
                solar_irradiance,
                sun_angular_radius,
                basis_right,
                basis_up,
                basis_forward,
                float(tangent_half_fov),
            )
            self.aerial_snapshot_altitude_m = camera_radius - planet_radius
            self._aerial_available = True
            self.aerial_rebuilds += 1

    def composite(
        self,
        scene_hdr,
        surface_id,
        position_view,
        hdr,
        planet_radius_m: float,
        camera_radius_m: float,
        sun_local: np.ndarray,
        solar_irradiance: tuple[float, float, float],
        sun_disk_radiance: tuple[float, float, float],
        sun_angular_radius_degrees: float,
        view_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
        tangent_half_fov: float,
        diagnostic_view: AtmosphereDiagnosticView = (
            AtmosphereDiagnosticView.COMPOSITE
        ),
    ) -> None:
        right, view_up, forward = view_basis
        sky_lut_camera_radius = float(camera_radius_m)
        if self.sky_snapshot_altitude_m is not None:
            sky_lut_camera_radius = (
                float(planet_radius_m) + self.sky_snapshot_altitude_m
            )
        self._composite(
            scene_hdr,
            surface_id,
            position_view,
            hdr,
            float(planet_radius_m),
            float(camera_radius_m),
            sky_lut_camera_radius,
            tuple(float(value) for value in sun_local),
            solar_irradiance,
            sun_disk_radiance,
            math.radians(sun_angular_radius_degrees),
            tuple(float(value) for value in right),
            tuple(float(value) for value in view_up),
            tuple(float(value) for value in forward),
            float(tangent_half_fov),
            int(diagnostic_view),
        )
