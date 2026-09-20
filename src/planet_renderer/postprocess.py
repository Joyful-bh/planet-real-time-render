"""Linear-HDR exposure, bloom, tone mapping and display encoding."""

from dataclasses import dataclass
import math
from typing import Mapping

import taichi as ti


@dataclass(frozen=True)
class PostprocessConfig:
    """Display parameters kept separate from physical scene radiance."""

    exposure_ev: float = 0.0
    bloom_threshold: float = 1.0
    bloom_knee: float = 0.5
    bloom_clamp: float = 8.0
    bloom_strength: float = 0.08
    bloom_passes: int = 4

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.exposure_ev)
            or not -20.0 <= self.exposure_ev <= 20.0
        ):
            raise ValueError("postprocess.exposure_ev must be finite and in [-20, 20]")
        finite_non_negative = {
            "bloom_threshold": self.bloom_threshold,
            "bloom_knee": self.bloom_knee,
            "bloom_clamp": self.bloom_clamp,
            "bloom_strength": self.bloom_strength,
        }
        for name, value in finite_non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"postprocess.{name} must be finite and non-negative")
        if not 0 <= self.bloom_passes <= 8:
            raise ValueError("postprocess.bloom_passes must be in 0..8")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PostprocessConfig":
        defaults = cls()
        return cls(
            exposure_ev=float(values.get("exposure_ev", defaults.exposure_ev)),
            bloom_threshold=float(
                values.get("bloom_threshold", defaults.bloom_threshold)
            ),
            bloom_knee=float(values.get("bloom_knee", defaults.bloom_knee)),
            bloom_clamp=float(values.get("bloom_clamp", defaults.bloom_clamp)),
            bloom_strength=float(
                values.get("bloom_strength", defaults.bloom_strength)
            ),
            bloom_passes=int(values.get("bloom_passes", defaults.bloom_passes)),
        )


@ti.func
def _display_from_exposed(value: ti.template()):
    """Apply the ACES fit and encode linear output as sRGB."""

    value = ti.max(value, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = ti.math.clamp(
        (value * (a * value + b)) / (value * (c * value + d) + e),
        0.0,
        1.0,
    )
    lo = mapped * 12.92
    hi = 1.055 * ti.pow(mapped, 1.0 / 2.4) - 0.055
    return ti.Vector(
        [lo[i] if mapped[i] <= 0.0031308 else hi[i] for i in ti.static(range(3))]
    )


@ti.data_oriented
class PostProcessor:
    """Half-resolution bloom followed by the sole display transform."""

    def __init__(self, width: int, height: int, config: PostprocessConfig):
        self.width = int(width)
        self.height = int(height)
        self.half_width = max((self.width + 1) // 2, 1)
        self.half_height = max((self.height + 1) // 2, 1)
        self.config = config
        shape = (self.half_width, self.half_height)
        self.bloom_a = ti.Vector.field(3, ti.f32, shape=shape)
        self.bloom_b = ti.Vector.field(3, ti.f32, shape=shape)

    @ti.func
    def _soft_threshold(self, value: ti.template()):
        brightness = ti.max(value.x, ti.max(value.y, value.z))
        threshold = ti.static(self.config.bloom_threshold)
        knee = ti.static(self.config.bloom_knee)
        soft = ti.math.clamp(brightness - threshold + knee, 0.0, 2.0 * knee)
        soft = soft * soft / ti.max(4.0 * knee, 1.0e-6)
        contribution = ti.max(brightness - threshold, soft)
        contribution /= ti.max(brightness, 1.0e-6)
        result = value * contribution
        result_peak = ti.max(result.x, ti.max(result.y, result.z))
        # A physically bright solar texel can be tens of thousands of display
        # units. A small finite blur kernel cannot represent the corresponding
        # long PSF tail: feeding that value through unchanged merely saturates
        # its rectangular support. Preserve chromaticity but cap the bright
        # pass to the range this realtime kernel can reconstruct smoothly.
        result *= ti.min(
            1.0,
            ti.static(self.config.bloom_clamp) / ti.max(result_peak, 1.0e-6),
        )
        return result

    @ti.kernel
    def _extract(self, hdr: ti.template(), exposure_multiplier: ti.f32):
        for x, y in self.bloom_a:
            value = ti.Vector.zero(ti.f32, 3)
            for ox, oy in ti.static(ti.ndrange(2, 2)):
                sx = ti.min(x * 2 + ox, self.width - 1)
                sy = ti.min(y * 2 + oy, self.height - 1)
                value += hdr[sx, sy]
            value *= 0.25 * exposure_multiplier
            self.bloom_a[x, y] = self._soft_threshold(value)

    @ti.kernel
    def _blur_horizontal(self):
        for x, y in self.bloom_b:
            color = self.bloom_a[x, y] * 0.38774
            color += self.bloom_a[ti.max(x - 1, 0), y] * 0.24477
            color += self.bloom_a[ti.min(x + 1, self.half_width - 1), y] * 0.24477
            color += self.bloom_a[ti.max(x - 2, 0), y] * 0.06136
            color += self.bloom_a[ti.min(x + 2, self.half_width - 1), y] * 0.06136
            self.bloom_b[x, y] = color

    @ti.kernel
    def _blur_vertical(self):
        for x, y in self.bloom_a:
            color = self.bloom_b[x, y] * 0.38774
            color += self.bloom_b[x, ti.max(y - 1, 0)] * 0.24477
            color += self.bloom_b[x, ti.min(y + 1, self.half_height - 1)] * 0.24477
            color += self.bloom_b[x, ti.max(y - 2, 0)] * 0.06136
            color += self.bloom_b[x, ti.min(y + 2, self.half_height - 1)] * 0.06136
            self.bloom_a[x, y] = color

    @ti.func
    def _sample_bloom(self, u, v):
        px = u * ti.static(self.half_width) - 0.5
        py = v * ti.static(self.half_height) - 0.5
        x0 = ti.math.clamp(ti.cast(ti.floor(px), ti.i32), 0, self.half_width - 1)
        y0 = ti.math.clamp(ti.cast(ti.floor(py), ti.i32), 0, self.half_height - 1)
        x1 = ti.min(x0 + 1, ti.static(self.half_width - 1))
        y1 = ti.min(y0 + 1, ti.static(self.half_height - 1))
        fx = ti.math.clamp(px - ti.floor(px), 0.0, 1.0)
        fy = ti.math.clamp(py - ti.floor(py), 0.0, 1.0)
        low = self.bloom_a[x0, y0] * (1.0 - fx) + self.bloom_a[x1, y0] * fx
        high = self.bloom_a[x0, y1] * (1.0 - fx) + self.bloom_a[x1, y1] * fx
        return low * (1.0 - fy) + high * fy

    @ti.kernel
    def _resolve(
        self,
        hdr: ti.template(),
        display: ti.template(),
        exposure_multiplier: ti.f32,
        bloom_strength: ti.f32,
        use_bloom: ti.i32,
        tone_map: ti.i32,
    ):
        for pixel in ti.grouped(display):
            value = hdr[pixel] * exposure_multiplier
            if use_bloom != 0:
                u = (ti.cast(pixel.x, ti.f32) + 0.5) / self.width
                v = (ti.cast(pixel.y, ti.f32) + 0.5) / self.height
                value += self._sample_bloom(u, v) * bloom_strength
            if tone_map != 0:
                display[pixel] = _display_from_exposed(value)
            else:
                display[pixel] = ti.math.clamp(value, 0.0, 1.0)

    def process(
        self,
        hdr,
        display,
        exposure_ev: float,
        *,
        bloom: bool,
        tone_map: bool,
    ) -> None:
        """Convert one HDR frame without ever modifying the HDR source."""

        # Raw transmittance/mask diagnostics deliberately bypass exposure so
        # their numeric [0, 1] meaning remains stable while the user changes EV.
        exposure_multiplier = 2.0 ** float(exposure_ev) if tone_map else 1.0
        use_bloom = bool(
            bloom
            and self.config.bloom_passes > 0
            and self.config.bloom_strength > 0.0
        )
        if use_bloom:
            self._extract(hdr, exposure_multiplier)
            for _ in range(self.config.bloom_passes):
                self._blur_horizontal()
                self._blur_vertical()
        self._resolve(
            hdr,
            display,
            exposure_multiplier,
            float(self.config.bloom_strength),
            int(use_bloom),
            int(tone_map),
        )
