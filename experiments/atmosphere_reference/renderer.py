"""参考大气原型的缓冲与 kernel 调度。"""

import math
import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from .atmosphere import shade_atmosphere
from .camera import generate_ray
from .config import RenderConfig
from .postprocess import display_transform


@dataclass(frozen=True)
class TimingResult:
    jit_seconds: float
    average_seconds: float
    frames_per_second: float


@ti.data_oriented
class SkyRenderer:
    """拥有线性 HDR 与显示缓冲的程序化天空渲染器。"""

    def __init__(self, config: RenderConfig):
        self.config = config
        shape = (config.width, config.height)
        self.hdr = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.display = ti.Vector.field(3, dtype=ti.f32, shape=shape)

    @ti.kernel
    def _render_kernel(
        self,
        yaw: ti.f32,
        pitch: ti.f32,
        fov: ti.f32,
        sun_direction: ti.types.vector(3, ti.f32),
        camera_position: ti.types.vector(3, ti.f32),
        planet_radius: ti.f32,
        atmosphere_radius: ti.f32,
        rayleigh_scale: ti.f32,
        mie_scale: ti.f32,
        beta_rayleigh: ti.types.vector(3, ti.f32),
        beta_mie_scattering: ti.types.vector(3, ti.f32),
        beta_mie_extinction: ti.types.vector(3, ti.f32),
        mie_g: ti.f32,
        solar_irradiance: ti.types.vector(3, ti.f32),
        sun_radiance: ti.types.vector(3, ti.f32),
        sun_radius: ti.f32,
        ground_albedo: ti.types.vector(3, ti.f32),
        exposure_ev: ti.f32,
        use_aces: ti.i32,
    ):
        for pixel in ti.grouped(self.hdr):
            ray = generate_ray(pixel, ti.Vector([self.config.width, self.config.height]), yaw, pitch, fov)
            color = shade_atmosphere(
                ray, camera_position, sun_direction, planet_radius, atmosphere_radius,
                rayleigh_scale, mie_scale, beta_rayleigh, beta_mie_scattering,
                beta_mie_extinction, mie_g, solar_irradiance, sun_radiance,
                sun_radius, ground_albedo,
            )
            self.hdr[pixel] = color
            self.display[pixel] = display_transform(color, exposure_ev, use_aces)

    def render(
        self,
        yaw_degrees: float | None = None,
        pitch_degrees: float | None = None,
        camera_local_km: tuple[float, float, float] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> None:
        c = self.config
        values = parameters or {}
        azimuth = math.radians(float(values.get("sun_azimuth_degrees", c.sun_azimuth_degrees)))
        elevation = math.radians(float(values.get("sun_elevation_degrees", c.sun_elevation_degrees)))
        sun_direction = (math.sin(azimuth) * math.cos(elevation), math.sin(elevation), math.cos(azimuth) * math.cos(elevation))
        local = camera_local_km or (0.0, c.camera_altitude_km, 0.0)
        mie_multiplier = float(values.get("mie_multiplier", 1.0))
        beta_mie_sca = tuple(x * mie_multiplier for x in c.beta_mie_scattering_per_km)
        beta_mie_ext = tuple(x * mie_multiplier for x in c.beta_mie_extinction_per_km)
        self._render_kernel(
            math.radians(c.yaw_degrees if yaw_degrees is None else yaw_degrees),
            math.radians(c.pitch_degrees if pitch_degrees is None else pitch_degrees),
            math.radians(float(values.get("vertical_fov_degrees", c.vertical_fov_degrees))),
            sun_direction,
            (local[0], c.planet_radius_km + local[1], local[2]),
            c.planet_radius_km, c.planet_radius_km + float(values.get("atmosphere_height_km", c.atmosphere_height_km)),
            float(values.get("rayleigh_scale_height_km", c.rayleigh_scale_height_km)),
            float(values.get("mie_scale_height_km", c.mie_scale_height_km)),
            c.beta_rayleigh_per_km, beta_mie_sca,
            beta_mie_ext, float(values.get("mie_g", c.mie_g)), c.solar_irradiance,
            c.sun_radiance, math.radians(c.sun_angular_radius_degrees),
            values.get("ground_albedo", c.ground_albedo),
            float(values.get("exposure_ev", c.exposure_ev)),
            int(bool(values.get("use_aces", c.tone_mapper == "aces"))),
        )

    def benchmark(self, repeats: int = 20) -> TimingResult:
        start = time.perf_counter()
        self.render()
        ti.sync()
        jit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(max(repeats, 1)):
            self.render()
        ti.sync()
        average = (time.perf_counter() - start) / max(repeats, 1)
        return TimingResult(jit_seconds, average, 1.0 / max(average, 1.0e-9))

    def display_numpy(self) -> np.ndarray:
        return self.display.to_numpy()

    def hdr_numpy(self) -> np.ndarray:
        return self.hdr.to_numpy()
