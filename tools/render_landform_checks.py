"""Render the standard M2 landform inspection viewpoints in one process.

This is a visual validation tool, not a benchmark.  It advances the bounded
terrain streamer with simulated time until each viewpoint has stable LOD and
residency, then writes the formal material view.  Running all viewpoints in a
single process amortizes Taichi JIT and keeps the comparison deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planet_renderer.cli import initial_state  # noqa: E402
from planet_renderer.config import load_config  # noqa: E402
from planet_renderer.ocean import opaque_surface_height_m  # noqa: E402
from planet_renderer.renderer import PlanetRenderer  # noqa: E402
from planet_renderer.terrain import CubeSphereTerrain, TerrainSettings  # noqa: E402
from planet_renderer.terrain_factory import create_terrain_model  # noqa: E402


VIEWPOINTS = (
    ("space_100km", 100_000.0, -70.0),
    ("altitude_20km", 20_000.0, -38.0),
    ("altitude_2km", 2_000.0, -18.0),
    ("near_surface_120m", 120.0, -8.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render M2 landform check images")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "planet.json")
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "vulkan", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "landform_checks",
    )
    parser.add_argument("--settle-max-frames", type=int, default=240)
    return parser.parse_args()


def terrain_settings(config) -> TerrainSettings:
    return TerrainSettings(
        patch_resolution=config.terrain_patch_resolution,
        max_level=config.terrain_max_level,
        split_sse_pixels=config.terrain_split_sse_pixels,
        merge_sse_pixels=config.terrain_merge_sse_pixels,
        max_desired_patches=config.terrain_max_desired_patches,
        max_gpu_patches=config.terrain_max_gpu_patches,
        build_budget_per_frame=config.terrain_build_budget_per_frame,
        upload_budget_per_frame=config.terrain_upload_budget_per_frame,
        lod_changes_per_update=config.terrain_lod_changes_per_update,
        cache_capacity=config.terrain_cache_capacity,
    )


def settle_viewpoint(
    terrain: CubeSphereTerrain,
    renderer: PlanetRenderer,
    camera,
    now: float,
    max_frames: int,
):
    stable_frames = 0
    latest = None
    for _ in range(max_frames):
        now += 0.11
        latest = terrain.update(
            camera,
            renderer.width,
            renderer.height,
            now=now,
        )
        renderer.apply_terrain_frame(latest)
        settled = (
            terrain.selector.last_changes == 0
            and latest.stats.requested_patches == 0
            and latest.stats.ready_patches == 0
            and not latest.uploads
        )
        stable_frames = stable_frames + 1 if settled else 0
        if stable_frames >= 3:
            break
    if latest is None:
        raise RuntimeError("terrain settling produced no frame")
    return now, latest


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    arch = {
        "auto": ti.gpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "cpu": ti.cpu,
    }[args.backend]
    ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)

    planet, camera, lighting, _ = initial_state(config)
    terrain = CubeSphereTerrain(
        planet,
        create_terrain_model(config.terrain),
        terrain_settings(config),
    )
    renderer = PlanetRenderer(
        config.width,
        config.height,
        terrain.height_model,
        config.atmosphere,
        config.postprocess,
        config.terrain_max_gpu_patches,
        config.terrain_patch_resolution,
        config.ocean,
    )
    renderer.debug_view = 0

    direction = np.array([0.0, 0.0, 1.0], np.float64)
    terrain_height = terrain.describe_surface(direction).height_m
    surface_height = opaque_surface_height_m(
        terrain_height,
        config.ocean.enabled,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulated_time = 1.0
    warmed_up = False

    for name, clearance_m, pitch_degrees in VIEWPOINTS:
        radial_altitude = clearance_m
        if clearance_m < 10_000.0:
            radial_altitude += surface_height
        camera.position_global = planet.surface_position(direction, radial_altitude)
        camera.yaw_degrees = 0.0
        camera.pitch_degrees = pitch_degrees

        simulated_time, frame = settle_viewpoint(
            terrain,
            renderer,
            camera,
            simulated_time,
            args.settle_max_frames,
        )
        if not warmed_up:
            renderer.warmup_raster_paths(
                planet,
                camera,
                lighting.snapshot(),
                config.surface_albedo,
                config.postprocess.exposure_ev,
            )
            warmed_up = True
        renderer.render(
            planet,
            camera,
            lighting.snapshot(),
            config.surface_albedo,
            config.postprocess.exposure_ev,
        )
        ti.sync()

        output = args.output_dir / f"{name}.png"
        ti.tools.imwrite(renderer.display_numpy(), str(output))
        stats = frame.stats
        print(
            f"{name}: altitude={planet.altitude_m(camera.position_global):.1f} m, "
            f"LOD={stats.min_lod}..{stats.max_lod}, "
            f"desired/resident/render={stats.desired_patches}/"
            f"{stats.resident_patches}/{stats.render_patches}, output={output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
