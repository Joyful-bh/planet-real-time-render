"""参考大气原型配置加载与数值保护。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


def bounded(v: Any, lo: float, hi: float, name: str) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name} 必须是有限数值")
    return min(max(x, lo), hi)


def color(v: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError(f"{name} 必须是三个非负数值")
    return tuple(bounded(x, 0, 100000, name) for x in v)  # type: ignore[return-value]


@dataclass(frozen=True)
class RenderConfig:
    width: int; height: int
    yaw_degrees: float; pitch_degrees: float; vertical_fov_degrees: float; camera_altitude_km: float
    planet_radius_km: float; atmosphere_height_km: float
    rayleigh_scale_height_km: float; mie_scale_height_km: float
    beta_rayleigh_per_km: tuple[float, float, float]
    beta_mie_scattering_per_km: tuple[float, float, float]
    beta_mie_extinction_per_km: tuple[float, float, float]
    mie_g: float; ground_albedo: tuple[float, float, float]
    sun_azimuth_degrees: float; sun_elevation_degrees: float
    solar_irradiance: tuple[float, float, float]; sun_radiance: tuple[float, float, float]
    sun_angular_radius_degrees: float; exposure_ev: float; tone_mapper: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderConfig":
        r, c, a, s, p = (data.get(k, {}) for k in ("rendering", "camera", "atmosphere", "sun", "postprocess"))
        tone = str(p.get("tone_mapper", "aces")).lower()
        if tone not in {"aces", "reinhard"}:
            raise ValueError("postprocess.tone_mapper 必须是 aces 或 reinhard")
        return cls(
            int(bounded(r.get("width", 1280), 16, 16384, "width")), int(bounded(r.get("height", 720), 16, 16384, "height")),
            bounded(c.get("yaw_degrees", 0), -360, 360, "yaw"), bounded(c.get("pitch_degrees", 0), -89, 89, "pitch"),
            bounded(c.get("vertical_fov_degrees", 60), 10, 140, "fov"), bounded(c.get("altitude_km", 0.002), 0.0001, 1000, "altitude"),
            bounded(a.get("planet_radius_km", 6360), 100, 100000, "planet_radius"), bounded(a.get("height_km", 100), 1, 1000, "atmosphere_height"),
            bounded(a.get("rayleigh_scale_height_km", 8), 0.1, 100, "rayleigh_scale"), bounded(a.get("mie_scale_height_km", 1.2), 0.05, 50, "mie_scale"),
            color(a.get("beta_rayleigh_per_km", [0.005802, 0.013558, 0.0331]), "beta_rayleigh"),
            color(a.get("beta_mie_scattering_per_km", [0.003996]*3), "beta_mie_scattering"),
            color(a.get("beta_mie_extinction_per_km", [0.00444]*3), "beta_mie_extinction"),
            bounded(a.get("mie_g", 0.8), -0.95, 0.95, "mie_g"), color(a.get("ground_albedo", [0.12, 0.13, 0.14]), "ground_albedo"),
            bounded(s.get("azimuth_degrees", 0), -360, 360, "sun_azimuth"), bounded(s.get("elevation_degrees", 25), -90, 90, "sun_elevation"),
            color(s.get("solar_irradiance", [20]*3), "solar_irradiance"), color(s.get("radiance", [80, 75, 65]), "sun_radiance"),
            bounded(s.get("angular_radius_degrees", 0.266), 0.05, 5, "sun_radius"), bounded(p.get("exposure_ev", 0), -10, 10, "exposure"), tone,
        )


def load_config(path: Path) -> RenderConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到配置文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置 JSON 无效：{path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise ValueError("配置根节点必须是对象")
    return RenderConfig.from_dict(data)
