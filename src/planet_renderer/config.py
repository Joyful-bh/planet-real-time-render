"""M0 配置结构。"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class M0Config:
    width: int = 1280
    height: int = 720
    planet_radius_m: float = 6_360_000.0
    initial_altitude_m: float = 2.0
    yaw_degrees: float = 0.0
    pitch_degrees: float = -8.0
    vertical_fov_degrees: float = 60.0
    sun_azimuth_degrees: float = 25.0
    sun_elevation_degrees: float = 32.0
    sun_angular_radius_degrees: float = 0.266
    solar_irradiance: tuple[float, float, float] = (4.0, 3.9, 3.7)
    sun_disk_radiance: tuple[float, float, float] = (80.0, 74.0, 62.0)
    surface_albedo: tuple[float, float, float] = (0.16, 0.20, 0.12)
    exposure_ev: float = 0.0
    floating_origin_threshold_m: float = 10_000.0
    terrain_seed: int = 7
    terrain_patch_resolution: int = 12
    terrain_max_level: int = 16
    terrain_split_sse_pixels: float = 64.0
    terrain_merge_sse_pixels: float = 32.0
    terrain_max_desired_patches: int = 180
    terrain_max_gpu_patches: int = 256
    terrain_build_budget_per_frame: int = 8
    terrain_upload_budget_per_frame: int = 4
    terrain_lod_changes_per_update: int = 8
    terrain_cache_capacity: int = 1024


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} 必须包含三个数值")
    return tuple(float(x) for x in value)  # type: ignore[return-value]


def load_config(path: Path) -> M0Config:
    data = json.loads(path.read_text(encoding="utf-8"))
    r, p, c, l, t = (data.get(k, {}) for k in ("rendering", "planet", "camera", "lighting", "terrain"))
    config = M0Config(
        width=int(r.get("width", 1280)), height=int(r.get("height", 720)),
        planet_radius_m=float(p.get("radius_m", 6_360_000.0)),
        initial_altitude_m=float(c.get("initial_altitude_m", 2.0)),
        yaw_degrees=float(c.get("yaw_degrees", 0.0)), pitch_degrees=float(c.get("pitch_degrees", -8.0)),
        vertical_fov_degrees=float(c.get("vertical_fov_degrees", 60.0)),
        sun_azimuth_degrees=float(l.get("sun_azimuth_degrees", 25.0)),
        sun_elevation_degrees=float(l.get("sun_elevation_degrees", 32.0)),
        sun_angular_radius_degrees=float(l.get("sun_angular_radius_degrees", 0.266)),
        solar_irradiance=_triple(l.get("solar_irradiance", [4.0, 3.9, 3.7]), "solar_irradiance"),
        sun_disk_radiance=_triple(l.get("sun_disk_radiance", [80.0, 74.0, 62.0]), "sun_disk_radiance"),
        surface_albedo=_triple(p.get("surface_albedo", [0.16, 0.20, 0.12]), "surface_albedo"),
        exposure_ev=float(r.get("exposure_ev", 0.0)),
        floating_origin_threshold_m=float(c.get("floating_origin_threshold_m", 10_000.0)),
        terrain_seed=int(t.get("seed", 7)),
        terrain_patch_resolution=int(t.get("patch_resolution", 12)),
        terrain_max_level=int(t.get("max_level", 16)),
        terrain_split_sse_pixels=float(t.get("split_sse_pixels", t.get("target_error_pixels", 64.0))),
        terrain_merge_sse_pixels=float(t.get("merge_sse_pixels", 32.0)),
        terrain_max_desired_patches=int(t.get("max_desired_patches", 180)),
        terrain_max_gpu_patches=int(t.get("max_gpu_patches", 256)),
        terrain_build_budget_per_frame=int(t.get("build_budget_per_frame", 8)),
        terrain_upload_budget_per_frame=int(t.get("upload_budget_per_frame", 4)),
        terrain_lod_changes_per_update=int(t.get("lod_changes_per_update", 8)),
        terrain_cache_capacity=int(t.get("cache_capacity", 1024)),
    )
    if not 16 <= config.width <= 16384 or not 16 <= config.height <= 16384:
        raise ValueError("分辨率必须在 16..16384 范围")
    if config.planet_radius_m <= 1.0 or config.initial_altitude_m < 0.0:
        raise ValueError("行星半径必须为正且初始高度不能为负")
    if not 10.0 <= config.vertical_fov_degrees <= 140.0:
        raise ValueError("垂直视场角必须在 10..140 度")
    if not 2 <= config.terrain_patch_resolution <= 64 or not 6 <= config.terrain_max_desired_patches <= config.terrain_max_gpu_patches:
        raise ValueError("terrain patch_resolution 或 patch 容量超出范围")
    if not 0.0 < config.terrain_merge_sse_pixels < config.terrain_split_sse_pixels:
        raise ValueError("terrain merge_sse_pixels 必须小于 split_sse_pixels")
    return config
