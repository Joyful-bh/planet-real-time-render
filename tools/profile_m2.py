"""Profile M2 terrain streaming and rasterization without opening a window.

The script deliberately synchronizes after terrain upload and after rendering.
This perturbs throughput, but separates CPU dispatch time from completed GPU
work and makes movement hitches attributable.  Use preview FPS for final UX.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planet_renderer.cli import initial_state  # noqa: E402
from planet_renderer.config import load_config  # noqa: E402
from planet_renderer.ocean import opaque_surface_height_m  # noqa: E402
from planet_renderer.renderer import PlanetRenderer  # noqa: E402
from planet_renderer.terrain import CubeSphereTerrain  # noqa: E402
from planet_renderer.terrain import TerrainSettings
from planet_renderer.terrain_factory import create_terrain_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2 terrain/render bottleneck profiler"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "planet.json")
    parser.add_argument(
        "--backend", choices=("auto", "cuda", "vulkan", "cpu"), default="auto"
    )
    parser.add_argument(
        "--scenario", choices=("stable", "move", "both"), default="both"
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--settle-max-frames",
        type=int,
        default=600,
        help="Maximum extra frames used to finish LOD streaming before measurement",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--movement-step-m", type=float, default=25.0)
    parser.add_argument(
        "--terrain-update-interval-frames",
        type=int,
        help="Override the preview terrain cadence; defaults to the config value",
    )
    parser.add_argument(
        "--altitude-m",
        type=float,
        help="Set the initial radial altitude for a targeted flight profile",
    )
    parser.add_argument(
        "--width", type=int, help="Override render width for scaling experiments"
    )
    parser.add_argument(
        "--height", type=int, help="Override render height for scaling experiments"
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable summary")
    parser.add_argument(
        "--python-profile",
        action="store_true",
        help="Print top cumulative Python call sites for measured frames",
    )
    return parser.parse_args()


def percentile_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, np.float64)
    return {
        "mean_ms": float(data.mean()),
        "p50_ms": float(np.percentile(data, 50)),
        "p95_ms": float(np.percentile(data, 95)),
        "p99_ms": float(np.percentile(data, 99)),
        "max_ms": float(data.max()),
    }


def make_scene(args: argparse.Namespace):
    config = load_config(args.config)
    arch = {"auto": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[
        args.backend
    ]
    ti.init(arch=arch, default_fp=ti.f32, offline_cache=True, kernel_profiler=True)
    planet, camera, lighting, _ = initial_state(config)
    settings = TerrainSettings(
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
    terrain = CubeSphereTerrain(
        planet,
        create_terrain_model(config.terrain),
        settings,
    )
    direction = camera.position_global / np.linalg.norm(camera.position_global)
    terrain_height = terrain.describe_surface(direction).height_m
    active_surface_height = opaque_surface_height_m(
        terrain_height,
        config.ocean.enabled,
    )
    camera.position_global = planet.surface_position(
        direction, active_surface_height + max(config.initial_altitude_m, 2.0)
    )
    if args.altitude_m is not None:
        camera.position_global = planet.surface_position(
            direction, max(args.altitude_m, 2.0)
        )
    width, height = args.width or config.width, args.height or config.height
    renderer = PlanetRenderer(
        width,
        height,
        terrain.height_model,
        config.atmosphere,
        config.postprocess,
        settings.max_gpu_patches,
        settings.patch_resolution,
        config.ocean,
    )
    return config, planet, camera, lighting, terrain, renderer


def execute_frame(
    config,
    planet,
    camera,
    lighting,
    terrain,
    renderer,
    simulated_time: float,
    movement_m: float,
    update_terrain: bool = True,
) -> dict[str, float]:
    if movement_m:
        camera.move_local(planet, movement_m, 0.0, 0.0)
    ti.sync()
    frame_start = time.perf_counter()
    start = frame_start
    terrain_frame = None
    if update_terrain:
        terrain_frame = terrain.update(
            camera,
            renderer.width,
            renderer.height,
            now=simulated_time,
        )
    terrain_dispatch_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    if terrain_frame is not None:
        renderer.apply_terrain_frame(terrain_frame)
    terrain_upload_dispatch_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    ti.sync()
    terrain_gpu_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    renderer.render(
        planet,
        camera,
        lighting.snapshot(),
        config.surface_albedo,
        config.postprocess.exposure_ev,
    )
    render_dispatch_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    ti.sync()
    render_gpu_ms = (time.perf_counter() - start) * 1000.0
    stats = terrain.tile_manager.stats
    return {
        "frame_ms": (time.perf_counter() - frame_start) * 1000.0,
        "terrain_dispatch_ms": terrain_dispatch_ms,
        "terrain_upload_dispatch_ms": terrain_upload_dispatch_ms,
        "terrain_gpu_ms": terrain_gpu_ms,
        "render_dispatch_ms": render_dispatch_ms,
        "render_gpu_ms": render_gpu_ms,
        "selection_ms": stats.selection_ms,
        "desired_patches": float(stats.desired_patches),
        "resident_patches": float(stats.resident_patches),
        "render_patches": float(stats.render_patches),
        "triangles": float(renderer.triangle_count),
        "active_slots": float(renderer.active_slot_count[None]),
        "clipped_triangles": float(
            min(renderer.clipped_count[None], renderer.raster_capacity)
        ),
        "max_tile_candidates": float(renderer.last_max_tile_candidates),
        "tile_overflow": float(renderer.last_tile_overflow),
    }


def run_scenario(name: str, args: argparse.Namespace, scene) -> dict[str, object]:
    config, planet, camera, lighting, terrain, renderer = scene
    simulated_time = 1.0
    for _ in range(max(args.warmup_frames, 1)):
        simulated_time += 1.0 / 60.0
        execute_frame(
            config, planet, camera, lighting, terrain, renderer, simulated_time, 0.0
        )
    settled_frames = 0
    for settled_frames in range(max(args.settle_max_frames, 1)):
        stats = terrain.tile_manager.stats
        if (
            terrain.selector.last_changes == 0
            and stats.requested_patches == 0
            and stats.ready_patches == 0
        ):
            break
        simulated_time += 1.0 / 60.0
        execute_frame(
            config, planet, camera, lighting, terrain, renderer, simulated_time, 0.0
        )
    else:
        print(
            f"Warning: terrain did not settle within {args.settle_max_frames} extra frames"
        )
    print(f"[{name}] settle frames after fixed warmup: {settled_frames}")
    ti.profiler.clear_kernel_profiler_info()
    rows: list[dict[str, float]] = []
    movement = args.movement_step_m if name == "move" else 0.0
    python_profiler = cProfile.Profile() if args.python_profile else None
    if python_profiler is not None:
        python_profiler.enable()
    update_interval = (
        args.terrain_update_interval_frames
        or config.terrain_update_interval_frames
    )
    for frame_index in range(max(args.frames, 1)):
        simulated_time += 1.0 / 60.0
        rows.append(
            execute_frame(
                config,
                planet,
                camera,
                lighting,
                terrain,
                renderer,
                simulated_time,
                movement,
                update_terrain=frame_index % update_interval == 0,
            )
        )
    if python_profiler is not None:
        python_profiler.disable()
        stream = io.StringIO()
        pstats.Stats(python_profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(35)
        print("\nPython profile (measured frames only):\n" + stream.getvalue())
    timing_keys = (
        "frame_ms",
        "terrain_dispatch_ms",
        "terrain_upload_dispatch_ms",
        "terrain_gpu_ms",
        "render_dispatch_ms",
        "render_gpu_ms",
        "selection_ms",
    )
    result: dict[str, object] = {
        key: percentile_summary([row[key] for row in rows]) for key in timing_keys
    }
    for key in (
        "desired_patches",
        "resident_patches",
        "render_patches",
        "triangles",
        "active_slots",
        "clipped_triangles",
        "max_tile_candidates",
        "tile_overflow",
    ):
        result[key] = {
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
    result["fps_from_mean"] = 1000.0 / max(result["frame_ms"]["mean_ms"], 1e-9)  # type: ignore[index]
    tile_counts = renderer.tile_counts.to_numpy().astype(np.float64).ravel()
    result["tile_candidates_last_frame"] = {
        "mean": float(tile_counts.mean()),
        "p95": float(np.percentile(tile_counts, 95)),
        "max": float(tile_counts.max()),
    }
    print(
        f"\n[{name}] {renderer.width}x{renderer.height}, {args.frames} measured frames"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nTaichi kernel profile (synchronized diagnostic run):")
    ti.profiler.print_kernel_profiler_info("count")
    return result


def main() -> int:
    args = parse_args()
    if (
        args.frames < 1
        or args.warmup_frames < 1
        or args.settle_max_frames < 1
        or args.movement_step_m < 0.0
        or (
            args.terrain_update_interval_frames is not None
            and args.terrain_update_interval_frames < 1
        )
    ):
        raise ValueError(
            "frames/warmup must be positive and movement-step-m must be non-negative"
        )
    names = ("stable", "move") if args.scenario == "both" else (args.scenario,)
    results: dict[str, object] = {}
    for index, name in enumerate(names):
        if index:
            ti.reset()
        scene = make_scene(args)
        results[name] = run_scenario(name, args, scene)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
