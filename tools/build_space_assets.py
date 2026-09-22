"""Build compact runtime star and Milky Way assets.

The tool is intentionally offline-only. Runtime code consumes NumPy caches and
therefore does not depend on Pillow or parse catalogue text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _parse_ra(value: str) -> float:
    hours, minutes, seconds = (float(part) for part in value.split())
    return math.radians((hours + minutes / 60.0 + seconds / 3600.0) * 15.0)


def _parse_dec(value: str) -> float:
    parts = value.split()
    sign = -1.0 if parts[0].startswith("-") else 1.0
    degrees = abs(float(parts[0]))
    minutes = float(parts[1])
    seconds = float(parts[2])
    return math.radians(sign * (degrees + minutes / 60.0 + seconds / 3600.0))


def _temperature_from_bv(colour_index: np.ndarray) -> np.ndarray:
    bv = np.clip(colour_index, -0.4, 2.0)
    return 4600.0 * (
        1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)
    )


def _blackbody_rgb(temperature_kelvin: np.ndarray) -> np.ndarray:
    wavelengths_m = np.asarray([680.0, 550.0, 440.0], dtype=np.float64) * 1.0e-9
    temperature = temperature_kelvin[:, None]
    c2 = 1.438776877e-2
    spectral = 1.0 / (
        wavelengths_m[None, :] ** 5
        * np.expm1(c2 / (wavelengths_m[None, :] * temperature))
    )
    spectral /= np.maximum(np.max(spectral, axis=1, keepdims=True), 1.0e-30)
    # Photographic star colours are much less saturated than monochromatic
    # wavelength samples. Retain temperature identity without neon points.
    return 0.62 + 0.38 * spectral


def build_star_catalog(source: Path, destination: Path) -> int:
    lines = source.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("recno\t")
    )
    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    directions: list[tuple[float, float, float]] = []
    magnitude: list[float] = []
    colour_index: list[float] = []
    hr_numbers: list[int] = []
    for row in reader:
        try:
            hr = int(row["HR"].strip())
            ra = _parse_ra(row["RAJ2000"].strip())
            dec = _parse_dec(row["DEJ2000"].strip())
            visual_magnitude = float(row["Vmag"].strip())
        except (KeyError, TypeError, ValueError):
            continue
        bv_text = row.get("B-V", "").strip()
        bv = float(bv_text) if bv_text else 0.65
        cos_dec = math.cos(dec)
        # Global celestial frame: +Y is ICRS north; RA grows from +X to +Z.
        directions.append(
            (cos_dec * math.cos(ra), math.sin(dec), cos_dec * math.sin(ra))
        )
        magnitude.append(visual_magnitude)
        colour_index.append(bv)
        hr_numbers.append(hr)

    direction_array = np.asarray(directions, dtype=np.float32)
    magnitude_array = np.asarray(magnitude, dtype=np.float32)
    bv_array = np.asarray(colour_index, dtype=np.float32)
    colours = _blackbody_rgb(_temperature_from_bv(bv_array)).astype(np.float32)
    order = np.argsort(magnitude_array, kind="stable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        directions=direction_array[order],
        colours=colours[order],
        magnitude=magnitude_array[order],
        bv=bv_array[order],
        hr=np.asarray(hr_numbers, dtype=np.int32)[order],
    )
    return int(direction_array.shape[0])


def _read_hipparcos(
    source: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions: list[tuple[float, float, float]] = []
    magnitude: list[float] = []
    colour_index: list[float] = []
    hip_numbers: list[int] = []
    with gzip.open(source, "rt", encoding="ascii") as catalogue:
        for line in catalogue:
            fields = line.rstrip("\n").split("|")
            try:
                hip = int(fields[1])
                visual_magnitude = float(fields[5])
                ra = math.radians(float(fields[8]))
                dec = math.radians(float(fields[9]))
            except (IndexError, ValueError):
                continue
            bv_text = fields[37].strip() if len(fields) > 37 else ""
            try:
                bv = float(bv_text) if bv_text else 0.65
            except ValueError:
                bv = 0.65
            cos_dec = math.cos(dec)
            directions.append(
                (cos_dec * math.cos(ra), math.sin(dec), cos_dec * math.sin(ra))
            )
            magnitude.append(visual_magnitude)
            colour_index.append(bv)
            hip_numbers.append(hip)

    direction_array = np.asarray(directions, dtype=np.float32)
    magnitude_array = np.asarray(magnitude, dtype=np.float32)
    bv_array = np.asarray(colour_index, dtype=np.float32)
    colours = _blackbody_rgb(_temperature_from_bv(bv_array)).astype(np.float32)
    return (
        direction_array,
        colours,
        magnitude_array,
        bv_array,
        np.asarray(hip_numbers, dtype=np.int32),
    )


def _encode_rgbe(image: np.ndarray) -> np.ndarray:
    maximum = np.max(image, axis=2)
    encoded = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    nonzero = maximum > 1.0e-30
    exponent = np.zeros(maximum.shape, dtype=np.int32)
    exponent[nonzero] = np.ceil(np.log2(maximum[nonzero])).astype(np.int32)
    scale = np.ones(maximum.shape, dtype=np.float32)
    scale[nonzero] = np.exp2(exponent[nonzero]).astype(np.float32)
    mantissa = image / scale[:, :, None]
    encoded[:, :, :3] = np.clip(
        np.rint(mantissa * 255.0),
        0.0,
        255.0,
    ).astype(np.uint8)
    encoded[:, :, 3][nonzero] = np.clip(
        exponent[nonzero] + 128,
        1,
        255,
    ).astype(np.uint8)
    return encoded


def _bake_faint_star_texture(
    directions: np.ndarray,
    colours: np.ndarray,
    magnitude: np.ndarray,
    destination: Path,
    preview: Path,
    width: int,
    height: int,
    point_magnitude_limit: float,
) -> int:
    selected = magnitude > point_magnitude_limit
    directions = directions[selected]
    colours = colours[selected]
    magnitude = magnitude[selected]
    image = np.zeros((height, width, 3), dtype=np.float32)

    longitude = np.arctan2(directions[:, 2], directions[:, 0])
    latitude = np.arcsin(np.clip(directions[:, 1], -1.0, 1.0))
    pixel_x = np.mod(longitude / (2.0 * np.pi), 1.0) * width - 0.5
    pixel_y = (0.5 - latitude / np.pi) * (height - 1)
    base_x = np.floor(pixel_x).astype(np.int32)
    base_y = np.floor(pixel_y).astype(np.int32)
    radiance = colours * np.power(10.0, -0.4 * magnitude[:, None])

    # A small footprint prevents bilinear lookup from dropping sub-texel stars
    # during camera motion. It represents the reconstruction filter, not a
    # physical angular diameter.
    sigma = 0.78
    for offset_y in range(-2, 3):
        sample_y = np.clip(base_y + offset_y, 0, height - 1)
        dy = sample_y.astype(np.float32) - pixel_y
        for offset_x in range(-2, 3):
            unwrapped_x = base_x + offset_x
            sample_x = np.mod(unwrapped_x, width)
            dx = unwrapped_x.astype(np.float32) - pixel_x
            weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            np.add.at(
                image,
                (sample_y, sample_x),
                radiance * weight[:, None],
            )

    encoded = _encode_rgbe(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        image_rgbe=encoded,
        projection=np.asarray("ICRS equirectangular"),
        point_magnitude_limit=np.asarray(point_magnitude_limit, dtype=np.float32),
        source_count=np.asarray(directions.shape[0], dtype=np.int32),
    )
    preview_linear = np.clip(image * 90.0, 0.0, 1.0)
    Image.fromarray(_linear_to_srgb(preview_linear), mode="RGB").save(
        preview,
        quality=92,
        optimize=True,
    )
    return int(directions.shape[0])


def build_hipparcos_assets(
    source: Path,
    catalogue_destination: Path,
    texture_destination: Path,
    preview: Path,
    width: int,
    height: int,
    point_magnitude_limit: float,
) -> tuple[int, int]:
    directions, colours, magnitude, bv, hip = _read_hipparcos(source)
    order = np.argsort(magnitude, kind="stable")
    catalogue_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        catalogue_destination,
        directions=directions[order],
        colours=colours[order],
        magnitude=magnitude[order],
        bv=bv[order],
        hip=hip[order],
    )
    faint_count = _bake_faint_star_texture(
        directions,
        colours,
        magnitude,
        texture_destination,
        preview,
        width,
        height,
        point_magnitude_limit,
    )
    return int(directions.shape[0]), faint_count


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    encoded = np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0.0, 255.0).astype(np.uint8)


def _resize_linear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    channels = []
    for channel in range(3):
        plane = Image.fromarray(image[:, :, channel], mode="F")
        resized = plane.resize((width, height), Image.Resampling.LANCZOS)
        channels.append(np.asarray(resized, dtype=np.float32))
    return np.stack(channels, axis=2)


def _box_blur_axis(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    padding = [(0, 0), (0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate(
        (np.zeros(zero_shape, dtype=np.float64), cumulative),
        axis=axis,
    )
    window = radius * 2 + 1
    upper = [slice(None), slice(None), slice(None)]
    lower = [slice(None), slice(None), slice(None)]
    upper[axis] = slice(window, None)
    lower[axis] = slice(None, -window)
    return (
        (cumulative[tuple(upper)] - cumulative[tuple(lower)]) / window
    ).astype(np.float32)


def _soft_background(image: np.ndarray) -> np.ndarray:
    result = image
    for _ in range(2):
        result = _box_blur_axis(result, 2, 1)
        result = _box_blur_axis(result, 2, 0)
    return result


def build_milky_way(
    source: Path,
    destination: Path,
    preview: Path,
    width: int,
    height: int,
) -> tuple[int, int]:
    source_image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    linear = _srgb_to_linear(source_image)
    black_level = np.percentile(linear.reshape(-1, 3), 1.0, axis=0)
    linear = np.maximum(linear - black_level[None, None, :], 0.0)
    linear = _resize_linear(linear, width, height)

    # Clamp compact positive high-frequency residuals against a small Gaussian
    # background. This removes photographed point stars while retaining broad
    # dust lanes, nebulae and the Galactic bulge for the separate BSC5 layer.
    background = _soft_background(linear)
    luminance = linear @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    background_luminance = background @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    point_ceiling = background_luminance + 0.018
    attenuation = np.minimum(1.0, point_ceiling / np.maximum(luminance, 1.0e-6))
    star_suppressed = linear * attenuation[:, :, None]
    encoded = _linear_to_srgb(star_suppressed)
    encoded = np.asarray(
        Image.fromarray(encoded, mode="RGB").filter(
            ImageFilter.MedianFilter(size=5)
        ),
        dtype=np.uint8,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        image_srgb=encoded,
        source_projection=np.asarray("galactic equirectangular"),
        credit=np.asarray("ESO/S. Brunier"),
    )
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="RGB").save(preview, quality=92, optimize=True)
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("assets/space/source"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/space/runtime"),
    )
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--star-width", type=int, default=4096)
    parser.add_argument("--star-height", type=int, default=2048)
    parser.add_argument("--point-magnitude-limit", type=float, default=6.5)
    args = parser.parse_args()
    if args.width < 64 or args.height < 32 or args.width != 2 * args.height:
        parser.error("Milky Way output must be a 2:1 image of at least 64x32")
    if (
        args.star_width < 256
        or args.star_height < 128
        or args.star_width != 2 * args.star_height
    ):
        parser.error("faint-star output must be a 2:1 image of at least 256x128")

    star_count, faint_count = build_hipparcos_assets(
        args.source_dir / "hip_main.dat.gz",
        args.output_dir / "hipparcos_stars.npz",
        args.output_dir
        / f"hipparcos_faint_{args.star_width}x{args.star_height}.npz",
        args.output_dir
        / f"hipparcos_faint_{args.star_width}x{args.star_height}_preview.jpg",
        args.star_width,
        args.star_height,
        args.point_magnitude_limit,
    )
    size = build_milky_way(
        args.source_dir / "eso0932a_large.jpg",
        args.output_dir / f"milky_way_{args.width}x{args.height}.npz",
        args.output_dir / f"milky_way_{args.width}x{args.height}_preview.jpg",
        args.width,
        args.height,
    )
    print(
        f"hipparcos={star_count}; faint={faint_count}; "
        f"milky_way={size[0]}x{size[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
