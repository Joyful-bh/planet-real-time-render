"""Catalogue-backed, camera-independent stellar background.

Bright-star directions live in the global planet frame while the much larger
faint catalogue is baked into an RGBE celestial texture. Atmospheric
extinction and visibility against the sky background are deliberately handled
by the atmosphere composite pass.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import taichi as ti


@dataclass(frozen=True)
class SpaceConfig:
    """Configuration for catalogue-backed space radiance and its fallback."""

    enabled: bool = True
    catalog_path: str | None = "assets/space/runtime/hipparcos_stars.npz"
    catalog_magnitude_limit: float = 6.5
    seed: int = 2027
    star_count: int = 8000
    radiance_scale: float = 0.08
    minimum_magnitude: float = -1.5
    maximum_magnitude: float = 6.5
    minimum_radius_pixels: float = 0.55
    maximum_radius_pixels: float = 1.45
    contrast_start: float = 2.0
    contrast_end: float = 8.0
    milky_way_enabled: bool = True
    milky_way_texture_path: str | None = (
        "assets/space/runtime/milky_way_2048x1024.npz"
    )
    milky_way_radiance_scale: float = 0.006
    milky_way_longitude_offset_degrees: float = 0.0
    milky_way_flip_longitude: bool = True
    faint_star_texture_enabled: bool = True
    faint_star_texture_path: str | None = (
        "assets/space/runtime/hipparcos_faint_4096x2048.npz"
    )
    faint_star_radiance_scale: float = 0.65

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SpaceConfig":
        config = cls(
            enabled=bool(values.get("enabled", True)),
            catalog_path=(
                str(
                    values.get(
                        "catalog_path",
                        "assets/space/runtime/hipparcos_stars.npz",
                    )
                )
                if values.get(
                    "catalog_path",
                    "assets/space/runtime/hipparcos_stars.npz",
                )
                is not None
                else None
            ),
            catalog_magnitude_limit=float(
                values.get("catalog_magnitude_limit", 6.5)
            ),
            seed=int(values.get("seed", 2027)),
            star_count=int(values.get("star_count", 8000)),
            radiance_scale=float(values.get("radiance_scale", 0.08)),
            minimum_magnitude=float(values.get("minimum_magnitude", -1.5)),
            maximum_magnitude=float(values.get("maximum_magnitude", 6.5)),
            minimum_radius_pixels=float(
                values.get("minimum_radius_pixels", 0.55)
            ),
            maximum_radius_pixels=float(
                values.get("maximum_radius_pixels", 1.45)
            ),
            contrast_start=float(values.get("contrast_start", 2.0)),
            contrast_end=float(values.get("contrast_end", 8.0)),
            milky_way_enabled=bool(values.get("milky_way_enabled", True)),
            milky_way_texture_path=(
                str(
                    values.get(
                        "milky_way_texture_path",
                        "assets/space/runtime/milky_way_2048x1024.npz",
                    )
                )
                if values.get(
                    "milky_way_texture_path",
                    "assets/space/runtime/milky_way_2048x1024.npz",
                )
                is not None
                else None
            ),
            milky_way_radiance_scale=float(
                values.get("milky_way_radiance_scale", 0.006)
            ),
            milky_way_longitude_offset_degrees=float(
                values.get("milky_way_longitude_offset_degrees", 0.0)
            ),
            milky_way_flip_longitude=bool(
                values.get("milky_way_flip_longitude", True)
            ),
            faint_star_texture_enabled=bool(
                values.get("faint_star_texture_enabled", True)
            ),
            faint_star_texture_path=(
                str(
                    values.get(
                        "faint_star_texture_path",
                        "assets/space/runtime/hipparcos_faint_4096x2048.npz",
                    )
                )
                if values.get(
                    "faint_star_texture_path",
                    "assets/space/runtime/hipparcos_faint_4096x2048.npz",
                )
                is not None
                else None
            ),
            faint_star_radiance_scale=float(
                values.get("faint_star_radiance_scale", 0.65)
            ),
        )
        if not 0 <= config.star_count <= 100_000:
            raise ValueError("space.star_count must be in 0..100000")
        if config.radiance_scale < 0.0:
            raise ValueError("space.radiance_scale must be non-negative")
        if not -2.0 <= config.catalog_magnitude_limit <= 12.0:
            raise ValueError("space.catalog_magnitude_limit must be in -2..12")
        if config.minimum_magnitude >= config.maximum_magnitude:
            raise ValueError(
                "space.minimum_magnitude must be less than maximum_magnitude"
            )
        if not 0.25 <= config.minimum_radius_pixels <= 4.0:
            raise ValueError("space.minimum_radius_pixels must be in 0.25..4")
        if not config.minimum_radius_pixels <= config.maximum_radius_pixels <= 4.0:
            raise ValueError(
                "space.maximum_radius_pixels must be in minimum_radius_pixels..4"
            )
        if not 0.0 <= config.contrast_start < config.contrast_end:
            raise ValueError("space contrast_start must be below contrast_end")
        if config.milky_way_radiance_scale < 0.0:
            raise ValueError(
                "space.milky_way_radiance_scale must be non-negative"
            )
        if config.faint_star_radiance_scale < 0.0:
            raise ValueError(
                "space.faint_star_radiance_scale must be non-negative"
            )
        return config


def _blackbody_rgb(temperature_kelvin: np.ndarray) -> np.ndarray:
    """Return normalized linear RGB-like black-body samples.

    Three broad wavelengths are sufficient here because stars are sub-pixel
    emitters; this avoids storing display-encoded palette colours in the HDR
    lighting path.
    """

    wavelengths_m = np.asarray([680.0, 550.0, 440.0], dtype=np.float64) * 1.0e-9
    temperature = temperature_kelvin[:, None]
    c2 = 1.438776877e-2
    spectral = 1.0 / (
        wavelengths_m[None, :] ** 5
        * np.expm1(c2 / (wavelengths_m[None, :] * temperature))
    )
    return spectral / np.maximum(np.max(spectral, axis=1, keepdims=True), 1.0e-30)


def generate_procedural_stars(
    config: SpaceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reproducible isotropic catalogue in global coordinates."""

    rng = np.random.default_rng(config.seed)
    count = config.star_count
    z = rng.uniform(-1.0, 1.0, count)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    directions = np.column_stack(
        (radial * np.cos(azimuth), z, radial * np.sin(azimuth))
    ).astype(np.float32)

    # A power distribution produces many faint stars without a conspicuous
    # hard brightness tier. Magnitude is converted to linear radiant flux.
    distribution = rng.random(count) ** 0.34
    magnitude = config.minimum_magnitude + (
        config.maximum_magnitude - config.minimum_magnitude
    ) * distribution
    flux = config.radiance_scale * np.power(10.0, -0.4 * magnitude)

    temperature = np.clip(rng.lognormal(np.log(5600.0), 0.28, count), 2800.0, 12000.0)
    colour = _blackbody_rgb(temperature)
    radiance = (colour * flux[:, None]).astype(np.float32)

    brightness = np.clip(
        (config.maximum_magnitude - magnitude)
        / (config.maximum_magnitude - config.minimum_magnitude),
        0.0,
        1.0,
    )
    radius = (
        config.minimum_radius_pixels
        + (config.maximum_radius_pixels - config.minimum_radius_pixels)
        * brightness**0.55
    ).astype(np.float32)
    return directions, radiance, radius


def load_star_catalog(
    path: Path,
    config: SpaceConfig,
    magnitude_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a preprocessed catalogue and apply the runtime magnitude limit."""

    with np.load(path, allow_pickle=False) as catalog:
        directions = np.asarray(catalog["directions"], dtype=np.float32)
        colours = np.asarray(catalog["colours"], dtype=np.float32)
        magnitude = np.asarray(catalog["magnitude"], dtype=np.float32)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or colours.shape != directions.shape
        or magnitude.shape != (directions.shape[0],)
    ):
        raise ValueError(f"invalid star catalogue arrays in {path}")
    limit = (
        config.catalog_magnitude_limit
        if magnitude_limit is None
        else float(magnitude_limit)
    )
    selected = np.isfinite(magnitude) & (magnitude <= limit)
    directions = directions[selected]
    colours = colours[selected]
    magnitude = magnitude[selected]
    if directions.shape[0] > 100_000:
        raise ValueError("runtime star catalogue exceeds 100000 entries")
    length = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(length, 1.0e-12)
    flux = config.radiance_scale * np.power(10.0, -0.4 * magnitude)
    radiance = colours * flux[:, None]
    brightness = np.clip(
        (limit - magnitude)
        / max(limit - config.minimum_magnitude, 1.0e-6),
        0.0,
        1.0,
    )
    radius = (
        config.minimum_radius_pixels
        + (config.maximum_radius_pixels - config.minimum_radius_pixels)
        * brightness**0.55
    )
    return (
        directions.astype(np.float32),
        radiance.astype(np.float32),
        radius.astype(np.float32),
    )


def load_milky_way_texture(path: Path) -> np.ndarray:
    """Load an sRGB-encoded compact texture and return linear RGB."""

    with np.load(path, allow_pickle=False) as texture:
        encoded = np.asarray(texture["image_srgb"], dtype=np.uint8)
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"invalid Milky Way texture in {path}")
    srgb = encoded.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return np.ascontiguousarray(linear, dtype=np.float32)


def load_faint_star_texture(path: Path) -> np.ndarray:
    """Load an RGBE-encoded linear-HDR equirectangular star texture."""

    with np.load(path, allow_pickle=False) as texture:
        encoded = np.asarray(texture["image_rgbe"], dtype=np.uint8)
    if encoded.ndim != 3 or encoded.shape[2] != 4:
        raise ValueError(f"invalid faint-star texture in {path}")
    return np.ascontiguousarray(encoded)


def faint_star_point_magnitude_limit(path: Path) -> float:
    """Return the point/texture split stored with a baked star texture."""

    with np.load(path, allow_pickle=False) as texture:
        return float(np.asarray(texture["point_magnitude_limit"]).item())


@ti.data_oriented
class SpaceRenderer:
    """Project catalogue stars and a low-frequency galaxy background."""

    def __init__(self, width: int, height: int, config: SpaceConfig):
        self.width = int(width)
        self.height = int(height)
        self.config = config
        faint_texture_path = (
            Path(config.faint_star_texture_path)
            if config.faint_star_texture_path
            else None
        )
        point_magnitude_limit = config.catalog_magnitude_limit
        if (
            config.faint_star_texture_enabled
            and faint_texture_path is not None
            and faint_texture_path.is_file()
        ):
            # The texture owns every source fainter than this boundary. Use
            # its baked split exactly so the point layer cannot leave a gap or
            # double-count stars when a preset retains an older limit.
            point_magnitude_limit = faint_star_point_magnitude_limit(
                faint_texture_path
            )
        catalog_path = Path(config.catalog_path) if config.catalog_path else None
        if catalog_path is not None and catalog_path.is_file():
            directions, radiance, radius = load_star_catalog(
                catalog_path,
                config,
                point_magnitude_limit,
            )
            self.catalog_source = str(catalog_path)
        else:
            directions, radiance, radius = generate_procedural_stars(config)
            self.catalog_source = "procedural"
        self.star_count = int(directions.shape[0])
        capacity = max(self.star_count, 1)
        self.directions = ti.Vector.field(3, ti.f32, shape=capacity)
        self.radiance = ti.Vector.field(3, ti.f32, shape=capacity)
        self.radius_pixels = ti.field(ti.f32, shape=capacity)
        self.hdr = ti.Vector.field(3, ti.f32, shape=(self.width, self.height))

        if self.star_count:
            self.directions.from_numpy(directions)
            self.radiance.from_numpy(radiance)
            self.radius_pixels.from_numpy(radius)

        texture_path = (
            Path(config.milky_way_texture_path)
            if config.milky_way_texture_path
            else None
        )
        if texture_path is not None and texture_path.is_file():
            milky_way = load_milky_way_texture(texture_path)
            self.milky_way_source = str(texture_path)
        else:
            milky_way = np.zeros((1, 1, 3), dtype=np.float32)
            self.milky_way_source = None
        self.milky_way_height = int(milky_way.shape[0])
        self.milky_way_width = int(milky_way.shape[1])
        self.milky_way = ti.Vector.field(
            3,
            ti.f32,
            shape=(self.milky_way_width, self.milky_way_height),
        )
        self.milky_way.from_numpy(np.transpose(milky_way, (1, 0, 2)))

        if faint_texture_path is not None and faint_texture_path.is_file():
            faint_stars = load_faint_star_texture(faint_texture_path)
            self.faint_star_source = str(faint_texture_path)
        else:
            faint_stars = np.zeros((1, 1, 4), dtype=np.uint8)
            self.faint_star_source = None
        self.faint_star_height = int(faint_stars.shape[0])
        self.faint_star_width = int(faint_stars.shape[1])
        self.faint_stars = ti.Vector.field(
            4,
            ti.u8,
            shape=(self.faint_star_width, self.faint_star_height),
        )
        self.faint_stars.from_numpy(np.transpose(faint_stars, (1, 0, 2)))

    @ti.kernel
    def _clear(self):
        for pixel in ti.grouped(self.hdr):
            self.hdr[pixel] = ti.Vector.zero(ti.f32, 3)

    @ti.func
    def _sample_milky_way(self, u: ti.f32, v: ti.f32):
        x = u * ti.cast(self.milky_way_width, ti.f32) - 0.5
        y = ti.math.clamp(v, 0.0, 1.0) * ti.cast(
            self.milky_way_height - 1,
            ti.f32,
        )
        x0_raw = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.math.clamp(
            ti.cast(ti.floor(y), ti.i32),
            0,
            self.milky_way_height - 1,
        )
        x0 = (x0_raw % self.milky_way_width + self.milky_way_width) % self.milky_way_width
        x1 = (x0 + 1) % self.milky_way_width
        y1 = ti.min(y0 + 1, self.milky_way_height - 1)
        fx = x - ti.floor(x)
        fy = y - ti.floor(y)
        lower = self.milky_way[x0, y0] * (1.0 - fx) + self.milky_way[x1, y0] * fx
        upper = self.milky_way[x0, y1] * (1.0 - fx) + self.milky_way[x1, y1] * fx
        return lower * (1.0 - fy) + upper * fy

    @ti.func
    def _decode_rgbe(self, value: ti.types.vector(4, ti.u8)):
        result = ti.Vector.zero(ti.f32, 3)
        exponent = ti.cast(value.w, ti.i32)
        if exponent > 0:
            scale = ti.exp(
                0.6931471805599453 * ti.cast(exponent - 128, ti.f32)
            ) / 255.0
            result = ti.Vector(
                [
                    ti.cast(value.x, ti.f32),
                    ti.cast(value.y, ti.f32),
                    ti.cast(value.z, ti.f32),
                ]
            ) * scale
        return result

    @ti.func
    def _sample_faint_stars(self, u: ti.f32, v: ti.f32):
        x = u * ti.cast(self.faint_star_width, ti.f32) - 0.5
        y = ti.math.clamp(v, 0.0, 1.0) * ti.cast(
            self.faint_star_height - 1,
            ti.f32,
        )
        x0_raw = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.math.clamp(
            ti.cast(ti.floor(y), ti.i32),
            0,
            self.faint_star_height - 1,
        )
        x0 = (
            (x0_raw % self.faint_star_width + self.faint_star_width)
            % self.faint_star_width
        )
        x1 = (x0 + 1) % self.faint_star_width
        y1 = ti.min(y0 + 1, self.faint_star_height - 1)
        fx = x - ti.floor(x)
        fy = y - ti.floor(y)
        lower = self._decode_rgbe(self.faint_stars[x0, y0]) * (1.0 - fx)
        lower += self._decode_rgbe(self.faint_stars[x1, y0]) * fx
        upper = self._decode_rgbe(self.faint_stars[x0, y1]) * (1.0 - fx)
        upper += self._decode_rgbe(self.faint_stars[x1, y1]) * fx
        return lower * (1.0 - fy) + upper * fy

    @ti.kernel
    def _render_backgrounds(
        self,
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
        tangent_half_fov: ti.f32,
        radiance_scale: ti.f32,
        longitude_offset: ti.f32,
        longitude_sign: ti.f32,
        faint_star_scale: ti.f32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        for pixel in ti.grouped(self.hdr):
            screen_u = (ti.cast(pixel.x, ti.f32) + 0.5) / self.width
            screen_v = (ti.cast(pixel.y, ti.f32) + 0.5) / self.height
            sx = (screen_u * 2.0 - 1.0) * aspect
            sy = screen_v * 2.0 - 1.0
            ray = (
                right * sx + view_up * sy + forward / tangent_half_fov
            ).normalized()

            # Global celestial axes are (ICRS x, ICRS z, ICRS y), keeping
            # declination north on global +Y. Reorder before the standard
            # ICRS-to-Galactic rotation.
            eq = ti.Vector([ray.x, ray.z, ray.y])
            galactic = ti.Vector(
                [
                    -0.0548755604 * eq.x
                    - 0.8734370902 * eq.y
                    - 0.4838350155 * eq.z,
                    0.4941094279 * eq.x
                    - 0.4448296300 * eq.y
                    + 0.7469822445 * eq.z,
                    -0.8676661490 * eq.x
                    - 0.1980763734 * eq.y
                    + 0.4559837762 * eq.z,
                ]
            )
            longitude = ti.atan2(galactic.y, galactic.x)
            latitude = ti.asin(ti.math.clamp(galactic.z, -1.0, 1.0))
            u = 0.5 + longitude_sign * longitude / (2.0 * np.pi) + longitude_offset
            u = u - ti.floor(u)
            v = 0.5 - latitude / np.pi
            background = self._sample_milky_way(u, v) * radiance_scale

            # Hipparcos uses the same global ICRS frame as the point catalogue.
            # The baked layer contains only stars fainter than the point split.
            equatorial_longitude = ti.atan2(ray.z, ray.x)
            star_u = equatorial_longitude / (2.0 * np.pi)
            star_u = star_u - ti.floor(star_u)
            star_v = 0.5 - ti.asin(
                ti.math.clamp(ray.y, -1.0, 1.0)
            ) / np.pi
            background += self._sample_faint_stars(
                star_u,
                star_v,
            ) * faint_star_scale
            self.hdr[pixel] = background

    @ti.kernel
    def _project(
        self,
        right: ti.types.vector(3, ti.f32),
        view_up: ti.types.vector(3, ti.f32),
        forward: ti.types.vector(3, ti.f32),
        tangent_half_fov: ti.f32,
    ):
        aspect = ti.cast(self.width, ti.f32) / self.height
        for index in range(self.star_count):
            direction = self.directions[index]
            view_x = direction.dot(right)
            view_y = direction.dot(view_up)
            view_z = direction.dot(forward)
            if view_z > 1.0e-5:
                ndc_x = view_x / (view_z * tangent_half_fov * aspect)
                ndc_y = view_y / (view_z * tangent_half_fov)
                if ti.abs(ndc_x) <= 1.01 and ti.abs(ndc_y) <= 1.01:
                    centre_x = (ndc_x * 0.5 + 0.5) * self.width - 0.5
                    centre_y = (ndc_y * 0.5 + 0.5) * self.height - 0.5
                    base_x = ti.cast(ti.floor(centre_x), ti.i32)
                    base_y = ti.cast(ti.floor(centre_y), ti.i32)
                    radius = self.radius_pixels[index]
                    sigma = ti.max(radius * 0.52, 0.32)
                    inverse_two_sigma_sq = 0.5 / (sigma * sigma)
                    for offset_x, offset_y in ti.static(ti.ndrange((-2, 3), (-2, 3))):
                        pixel_x = base_x + offset_x
                        pixel_y = base_y + offset_y
                        if (
                            0 <= pixel_x < self.width
                            and 0 <= pixel_y < self.height
                        ):
                            dx = ti.cast(pixel_x, ti.f32) - centre_x
                            dy = ti.cast(pixel_y, ti.f32) - centre_y
                            weight = ti.exp(
                                -(dx * dx + dy * dy) * inverse_two_sigma_sq
                            )
                            contribution = self.radiance[index] * weight
                            for channel in ti.static(range(3)):
                                ti.atomic_add(
                                    self.hdr[pixel_x, pixel_y][channel],
                                    contribution[channel],
                                )

    def render(
        self,
        view_basis_global: tuple[np.ndarray, np.ndarray, np.ndarray],
        tangent_half_fov: float,
    ) -> None:
        """Render stars; all inputs are global unit directions."""

        self._clear()
        if not self.config.enabled:
            return
        right, view_up, forward = view_basis_global
        basis_right = tuple(float(value) for value in right)
        basis_up = tuple(float(value) for value in view_up)
        basis_forward = tuple(float(value) for value in forward)
        milky_way_scale = 0.0
        if self.config.milky_way_enabled and self.milky_way_source is not None:
            milky_way_scale = self.config.milky_way_radiance_scale
        faint_star_scale = 0.0
        if (
            self.config.faint_star_texture_enabled
            and self.faint_star_source is not None
        ):
            faint_star_scale = self.config.faint_star_radiance_scale
        if milky_way_scale > 0.0 or faint_star_scale > 0.0:
            longitude_sign = -1.0 if self.config.milky_way_flip_longitude else 1.0
            self._render_backgrounds(
                basis_right,
                basis_up,
                basis_forward,
                float(tangent_half_fov),
                milky_way_scale,
                np.deg2rad(self.config.milky_way_longitude_offset_degrees)
                / (2.0 * np.pi),
                longitude_sign,
                faint_star_scale,
            )
        if self.star_count:
            self._project(
                basis_right,
                basis_up,
                basis_forward,
                float(tangent_half_fov),
            )


__all__ = [
    "SpaceConfig",
    "SpaceRenderer",
    "faint_star_point_magnitude_limit",
    "generate_procedural_stars",
    "load_faint_star_texture",
    "load_milky_way_texture",
    "load_star_catalog",
]
