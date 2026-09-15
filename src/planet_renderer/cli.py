"""M0 行星渲染器入口、离线输出与交互预览。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

from .camera import PlanetCamera
from .config import M0Config, load_config
from .lighting import LightingState, StaticLightingProvider
from .planet import FloatingOrigin, PlanetModel
from .renderer import PlanetRenderer
from .terrain import CubeSphereTerrain, ProceduralHeightSource, TerrainSettings


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi M0 解析行星渲染器")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "planet.json")
    parser.add_argument("--backend", choices=("auto", "cuda", "vulkan", "cpu"), default="auto")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-hdr", type=Path)
    parser.add_argument("--benchmark-frames", type=int, default=20)
    parser.add_argument("--altitude-m", type=float, help="覆盖初始径向高度，用于离线检查")
    parser.add_argument("--yaw-degrees", type=float, help="覆盖初始 yaw")
    parser.add_argument("--pitch-degrees", type=float, help="覆盖初始 pitch")
    return parser.parse_args(argv)


def initialize_taichi(backend: str) -> None:
    arch = {"auto": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[backend]
    ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)


def initial_state(config: M0Config):
    planet = PlanetModel(config.planet_radius_m)
    position = planet.surface_position(np.array([0.0, 0.0, 1.0]), config.initial_altitude_m)
    camera = PlanetCamera(position, config.yaw_degrees, config.pitch_degrees, config.vertical_fov_degrees)
    frame = planet.local_frame(position)
    azimuth, elevation = math.radians(config.sun_azimuth_degrees), math.radians(config.sun_elevation_degrees)
    sun_local = np.array([
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    ])
    lighting = LightingState(
        frame.local_to_global_direction(sun_local), config.sun_angular_radius_degrees,
        config.solar_irradiance, config.sun_disk_radiance,
    )
    origin = FloatingOrigin(position.copy(), config.floating_origin_threshold_m)
    return planet, camera, StaticLightingProvider(lighting), origin


def save_image(renderer: PlanetRenderer, path: Path | None, hdr_path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ti.tools.imwrite(renderer.display_numpy(), str(path))
        print(f"已保存图像：{path.resolve()}")
    if hdr_path is not None:
        hdr_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hdr_path, renderer.hdr_numpy())
        print(f"已保存线性 HDR：{hdr_path.resolve()}")


def run_preview(
    renderer: PlanetRenderer, config: M0Config, planet: PlanetModel,
    camera: PlanetCamera, provider: StaticLightingProvider, origin: FloatingOrigin, terrain: CubeSphereTerrain,
) -> None:
    window = ti.ui.Window("Taichi Planet Renderer - M2", (config.width, config.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    previous_mouse: tuple[float, float] | None = None
    mouse_sensitivity, speed_exponent = 180.0, 2.0
    exposure = config.exposure_ev
    last_time = time.perf_counter()
    while window.running:
        now = time.perf_counter()
        dt, last_time = min(now - last_time, 0.1), now
        if window.is_pressed(ti.ui.ESCAPE):
            break
        cursor = window.get_cursor_pos()
        over_panel = cursor[0] >= 0.70
        if window.is_pressed(ti.ui.LMB) and not over_panel:
            if previous_mouse is not None:
                camera.yaw_degrees += (cursor[0] - previous_mouse[0]) * mouse_sensitivity
                camera.pitch_degrees = float(np.clip(
                    camera.pitch_degrees + (cursor[1] - previous_mouse[1]) * mouse_sensitivity, -89.0, 89.0
                ))
            previous_mouse = cursor
        else:
            previous_mouse = None

        yaw = math.radians(camera.yaw_degrees)
        forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        motion = forward * (int(window.is_pressed("w")) - int(window.is_pressed("s")))
        motion += right * (int(window.is_pressed("d")) - int(window.is_pressed("a")))
        motion[1] += int(window.is_pressed(ti.ui.SPACE)) - int(window.is_pressed(ti.ui.SHIFT))
        length = float(np.linalg.norm(motion))
        if length > 0.0:
            distance = 10.0**speed_exponent * dt
            camera.move_local(planet, *(motion / length * distance))
        origin.update(camera.position_global)

        terrain_frame = terrain.update(camera, config.height, renderer)
        renderer.render(planet, camera, provider.snapshot(), config.surface_albedo, exposure)
        canvas.set_image(renderer.display)
        altitude = planet.altitude_m(camera.position_global)
        with gui.sub_window("M0 Planet", 0.705, 0.02, 0.28, 0.72) as panel:
            panel.text(f"Altitude: {altitude:,.2f} m")
            panel.text(f"Horizon: {planet.horizon_distance_m(altitude) / 1000.0:,.2f} km")
            panel.text(f"Origin revision: {origin.revision}")
            stats = terrain_frame.stats
            panel.text(f"LOD: {stats.min_lod}..{stats.max_lod}")
            panel.text(f"Patch D/R/V: {stats.desired_patches}/{stats.resident_patches}/{stats.render_patches}")
            panel.text(f"State requested/ready: {stats.requested_patches}/{stats.ready_patches}")
            panel.text(f"Select/build/upload: {stats.selection_ms:.2f}/{stats.build_ms:.2f}/{stats.upload_ms:.2f} ms")
            panel.text(f"Cache hit: {stats.cache_hit_rate*100:.1f}%")
            panel.text(("Surface: height" if renderer.debug_view == 0 else "Surface: LOD" if renderer.debug_view == 1 else "Surface: patch ID"))
            panel.text(f"Raster: {renderer.vertex_count:,} vertices / {renderer.triangle_count:,} triangles")
            relative = origin.relative_f32(camera.position_global)
            panel.text(f"Camera relative: {relative[0]:.1f}, {relative[1]:.1f}, {relative[2]:.1f}")
            speed_exponent = panel.slider_float("Speed log10(m/s)", speed_exponent, 0.0, 7.0)
            mouse_sensitivity = panel.slider_float("Mouse sensitivity", mouse_sensitivity, 30.0, 500.0)
            camera.vertical_fov_degrees = panel.slider_float("Vertical FOV", camera.vertical_fov_degrees, 20.0, 120.0)
            exposure = panel.slider_float("Exposure EV", exposure, -5.0, 5.0)
            if panel.button("Debug: Height"): renderer.debug_view = 0
            if panel.button("Debug: LOD"): renderer.debug_view = 1
            if panel.button("Debug: Patch ID"): renderer.debug_view = 2
            if panel.button("Surface (2 m)"):
                direction = camera.position_global / np.linalg.norm(camera.position_global)
                terrain_height = terrain.describe_surface(direction).height_m
                camera.position_global = planet.surface_position(direction, terrain_height + 2.0)
            if panel.button("High atmosphere (50 km)"):
                camera.position_global = planet.surface_position(camera.position_global, 50_000.0)
            if panel.button("Space (2,000 km)"):
                camera.position_global = planet.surface_position(camera.position_global, 2_000_000.0)
            panel.text("WASD | Space/Shift | hold LMB")
        window.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        initialize_taichi(args.backend)
        planet, camera, provider, origin = initial_state(config)
        if args.altitude_m is not None:
            camera.position_global = planet.surface_position(camera.position_global, max(args.altitude_m, 0.5))
            origin.origin_global = camera.position_global.copy()
        if args.yaw_degrees is not None:
            camera.yaw_degrees = args.yaw_degrees
        if args.pitch_degrees is not None:
            camera.pitch_degrees = float(np.clip(args.pitch_degrees, -89.0, 89.0))
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
            ProceduralHeightSource(config.terrain_seed),
            settings
        )
        requested_altitude = planet.altitude_m(camera.position_global)
        if requested_altitude < 20_000.0:
            direction = camera.position_global / np.linalg.norm(camera.position_global)
            terrain_height = terrain.describe_surface(direction).height_m
            camera.position_global = planet.surface_position(direction, terrain_height + max(requested_altitude, 2.0))
            origin.origin_global = camera.position_global.copy()
        renderer = PlanetRenderer(config.width, config.height, config.terrain_max_gpu_patches, config.terrain_patch_resolution, terrain.height_provider)
        # 两个固定预算 bootstrap tick 使六个根 patch 可作为初始 fallback；不等待细分完成。
        terrain.update(camera, config.height, renderer)
        terrain.update(camera, config.height, renderer)
        render_call = lambda: renderer.render(planet, camera, provider.snapshot(), config.surface_albedo, config.exposure_ev)
        timing = renderer.benchmark(render_call, args.benchmark_frames)
        raster_stats = renderer.stats()
        print(f"后端：{ti.lang.impl.current_cfg().arch}")
        print(f"首次渲染（含 JIT）：{timing.jit_seconds * 1000:.2f} ms")
        print(f"稳定帧时间：{timing.average_seconds * 1000:.2f} ms ({timing.frames_per_second:.1f} FPS)")
        print(f"相机高度：{planet.altitude_m(camera.position_global):.3f} m")
        print(f"解析地平线距离：{planet.horizon_distance_m(planet.altitude_m(camera.position_global)):.3f} m")
        print(f"光栅统计：{raster_stats.patches} 分块，{raster_stats.vertices} 顶点，{raster_stats.triangles} 三角形，tile 溢出 {raster_stats.tile_overflow}")
        output = args.output
        if output is None and not args.preview:
            output = ROOT / "output" / "m0_planet.png"
        save_image(renderer, output, args.save_hdr)
        if args.preview:
            run_preview(renderer, config, planet, camera, provider, origin, terrain)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
