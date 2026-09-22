"""Configuration contract for the spherical atmosphere.

All lengths are metres and all scattering/extinction coefficients are inverse
metres.  The planet radius deliberately does not live in this object: the
atmosphere always consumes the active :class:`PlanetModel` so the two shells
cannot silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


Rgb = tuple[float, float, float]


def _rgb(value: object, name: str, default: Rgb) -> Rgb:
    source = default if value is None else value
    if not isinstance(source, (list, tuple)) or len(source) != 3:
        raise ValueError(f"atmosphere.{name} must contain three values")
    result = tuple(float(component) for component in source)
    if not all(math.isfinite(component) and component >= 0.0 for component in result):
        raise ValueError(f"atmosphere.{name} must be finite and non-negative")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class AtmosphereConfig:
    """Optical parameters and LUT quality for one spherical atmosphere."""

    top_altitude_m: float = 100_000.0
    rayleigh_scattering_per_m: Rgb = (5.802e-6, 13.558e-6, 33.100e-6)
    rayleigh_scale_height_m: float = 8_000.0
    mie_scattering_per_m: Rgb = (3.996e-6, 3.996e-6, 3.996e-6)
    mie_extinction_per_m: Rgb = (4.440e-6, 4.440e-6, 4.440e-6)
    mie_scale_height_m: float = 1_200.0
    mie_phase_g: float = 0.80
    absorption_extinction_per_m: Rgb = (0.650e-6, 1.881e-6, 0.085e-6)
    absorption_peak_altitude_m: float = 25_000.0
    absorption_half_width_m: float = 15_000.0
    ground_albedo: Rgb = (0.10, 0.10, 0.10)
    transmittance_lut_width: int = 256
    transmittance_lut_height: int = 64
    multi_scattering_lut_width: int = 32
    multi_scattering_lut_height: int = 32
    sky_view_lut_width: int = 192
    sky_view_lut_height: int = 256
    aerial_lut_width: int = 32
    aerial_lut_height: int = 18
    aerial_lut_depth: int = 32
    transmittance_steps: int = 40
    multi_scattering_directions: int = 16
    multi_scattering_steps: int = 12
    sky_view_steps: int = 24
    aerial_steps_per_slice: int = 2
    aerial_horizon_raymarch_steps: int = 12
    aerial_terminator_substeps: int = 8

    def __post_init__(self) -> None:
        coefficient_groups = {
            "rayleigh_scattering_per_m": self.rayleigh_scattering_per_m,
            "mie_scattering_per_m": self.mie_scattering_per_m,
            "mie_extinction_per_m": self.mie_extinction_per_m,
            "absorption_extinction_per_m": self.absorption_extinction_per_m,
            "ground_albedo": self.ground_albedo,
        }
        for name, values in coefficient_groups.items():
            if len(values) != 3 or not all(
                math.isfinite(value) and value >= 0.0 for value in values
            ):
                raise ValueError(
                    f"atmosphere.{name} must contain three non-negative values"
                )
        if any(value > 1.0 for value in self.ground_albedo):
            raise ValueError("atmosphere.ground_albedo must remain in [0, 1]")
        positive = {
            "top_altitude_m": self.top_altitude_m,
            "rayleigh_scale_height_m": self.rayleigh_scale_height_m,
            "mie_scale_height_m": self.mie_scale_height_m,
            "absorption_half_width_m": self.absorption_half_width_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"atmosphere.{name} must be positive and finite")
        if not 0.0 <= self.absorption_peak_altitude_m <= self.top_altitude_m:
            raise ValueError("atmosphere absorption peak must lie inside the shell")
        if not -0.99 < self.mie_phase_g < 0.99:
            raise ValueError("atmosphere.mie_phase_g must be in (-0.99, 0.99)")
        for scattering, extinction in zip(
            self.mie_scattering_per_m,
            self.mie_extinction_per_m,
        ):
            if scattering > extinction:
                raise ValueError("Mie scattering cannot exceed Mie extinction")
        if not 16 <= self.transmittance_lut_width <= 2048:
            raise ValueError("transmittance LUT width must be in 16..2048")
        if not 16 <= self.transmittance_lut_height <= 1024:
            raise ValueError("transmittance LUT height must be in 16..1024")
        if not 16 <= self.sky_view_lut_width <= 2048:
            raise ValueError("sky-view LUT width must be in 16..2048")
        if not 16 <= self.sky_view_lut_height <= 2048:
            raise ValueError("sky-view LUT height must be in 16..2048")
        if not 8 <= self.multi_scattering_lut_width <= 256:
            raise ValueError("multi-scattering LUT width must be in 8..256")
        if not 8 <= self.multi_scattering_lut_height <= 256:
            raise ValueError("multi-scattering LUT height must be in 8..256")
        if not 4 <= self.aerial_lut_width <= 256:
            raise ValueError("aerial LUT width must be in 4..256")
        if not 4 <= self.aerial_lut_height <= 256:
            raise ValueError("aerial LUT height must be in 4..256")
        if not 8 <= self.aerial_lut_depth <= 256:
            raise ValueError("aerial LUT depth must be in 8..256")
        if not 4 <= self.transmittance_steps <= 256:
            raise ValueError("transmittance_steps must be in 4..256")
        if not 4 <= self.multi_scattering_directions <= 128:
            raise ValueError("multi_scattering_directions must be in 4..128")
        if not 4 <= self.multi_scattering_steps <= 128:
            raise ValueError("multi_scattering_steps must be in 4..128")
        if not 4 <= self.sky_view_steps <= 256:
            raise ValueError("sky_view_steps must be in 4..256")
        if not 1 <= self.aerial_steps_per_slice <= 16:
            raise ValueError("aerial_steps_per_slice must be in 1..16")
        if not 4 <= self.aerial_horizon_raymarch_steps <= 64:
            raise ValueError("aerial_horizon_raymarch_steps must be in 4..64")
        if not 2 <= self.aerial_terminator_substeps <= 16:
            raise ValueError("aerial_terminator_substeps must be in 2..16")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "AtmosphereConfig":
        defaults = cls()
        return cls(
            top_altitude_m=float(
                values.get("top_altitude_m", defaults.top_altitude_m)
            ),
            rayleigh_scattering_per_m=_rgb(
                values.get("rayleigh_scattering_per_m"),
                "rayleigh_scattering_per_m",
                defaults.rayleigh_scattering_per_m,
            ),
            rayleigh_scale_height_m=float(
                values.get(
                    "rayleigh_scale_height_m",
                    defaults.rayleigh_scale_height_m,
                )
            ),
            mie_scattering_per_m=_rgb(
                values.get("mie_scattering_per_m"),
                "mie_scattering_per_m",
                defaults.mie_scattering_per_m,
            ),
            mie_extinction_per_m=_rgb(
                values.get("mie_extinction_per_m"),
                "mie_extinction_per_m",
                defaults.mie_extinction_per_m,
            ),
            mie_scale_height_m=float(
                values.get("mie_scale_height_m", defaults.mie_scale_height_m)
            ),
            mie_phase_g=float(values.get("mie_phase_g", defaults.mie_phase_g)),
            absorption_extinction_per_m=_rgb(
                values.get("absorption_extinction_per_m"),
                "absorption_extinction_per_m",
                defaults.absorption_extinction_per_m,
            ),
            absorption_peak_altitude_m=float(
                values.get(
                    "absorption_peak_altitude_m",
                    defaults.absorption_peak_altitude_m,
                )
            ),
            absorption_half_width_m=float(
                values.get(
                    "absorption_half_width_m",
                    defaults.absorption_half_width_m,
                )
            ),
            ground_albedo=_rgb(
                values.get("ground_albedo"),
                "ground_albedo",
                defaults.ground_albedo,
            ),
            transmittance_lut_width=int(
                values.get(
                    "transmittance_lut_width",
                    defaults.transmittance_lut_width,
                )
            ),
            transmittance_lut_height=int(
                values.get(
                    "transmittance_lut_height",
                    defaults.transmittance_lut_height,
                )
            ),
            multi_scattering_lut_width=int(
                values.get(
                    "multi_scattering_lut_width",
                    defaults.multi_scattering_lut_width,
                )
            ),
            multi_scattering_lut_height=int(
                values.get(
                    "multi_scattering_lut_height",
                    defaults.multi_scattering_lut_height,
                )
            ),
            sky_view_lut_width=int(
                values.get("sky_view_lut_width", defaults.sky_view_lut_width)
            ),
            sky_view_lut_height=int(
                values.get("sky_view_lut_height", defaults.sky_view_lut_height)
            ),
            aerial_lut_width=int(
                values.get("aerial_lut_width", defaults.aerial_lut_width)
            ),
            aerial_lut_height=int(
                values.get("aerial_lut_height", defaults.aerial_lut_height)
            ),
            aerial_lut_depth=int(
                values.get("aerial_lut_depth", defaults.aerial_lut_depth)
            ),
            transmittance_steps=int(
                values.get("transmittance_steps", defaults.transmittance_steps)
            ),
            multi_scattering_directions=int(
                values.get(
                    "multi_scattering_directions",
                    defaults.multi_scattering_directions,
                )
            ),
            multi_scattering_steps=int(
                values.get(
                    "multi_scattering_steps",
                    defaults.multi_scattering_steps,
                )
            ),
            sky_view_steps=int(
                values.get("sky_view_steps", defaults.sky_view_steps)
            ),
            aerial_steps_per_slice=int(
                values.get(
                    "aerial_steps_per_slice",
                    defaults.aerial_steps_per_slice,
                )
            ),
            aerial_horizon_raymarch_steps=int(
                values.get(
                    "aerial_horizon_raymarch_steps",
                    defaults.aerial_horizon_raymarch_steps,
                )
            ),
            aerial_terminator_substeps=int(
                values.get(
                    "aerial_terminator_substeps",
                    defaults.aerial_terminator_substeps,
                )
            ),
        )
