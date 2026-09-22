"""M0 行星渲染器入口、离线输出与交互预览。"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import math
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

from .atmosphere import AtmosphereDiagnosticView
from .camera import PlanetCamera
from .config import M0Config, load_config
from .lighting import LightingState, StaticLightingProvider
from .ocean import opaque_surface_height_m
from .planet import FloatingOrigin, PlanetModel
from .renderer import PlanetRenderer
from .terrain import CubeSphereTerrain, TerrainSettings
from .terrain_factory import create_terrain_model

ROOT = Path(__file__).resolve().parents[2]
_ATMOSPHERE_DIAGNOSTIC_BUTTONS = (
    ("Atmo: Composite", AtmosphereDiagnosticView.COMPOSITE),
    ("Atmo: Sky-view", AtmosphereDiagnosticView.SKY_VIEW),
    ("Atmo: Camera T", AtmosphereDiagnosticView.CAMERA_TRANSMITTANCE),
    ("Atmo: Transmittance LUT", AtmosphereDiagnosticView.TRANSMITTANCE_LUT),
    (
        "Atmo: Multi-scattering LUT",
        AtmosphereDiagnosticView.MULTI_SCATTERING_LUT,
    ),
    ("Atmo: Aerial scattering", AtmosphereDiagnosticView.AERIAL_SCATTERING),
    ("Aerial: Froxel only", AtmosphereDiagnosticView.AERIAL_FROXEL_ONLY),
    ("Aerial: Direct only", AtmosphereDiagnosticView.AERIAL_DIRECT_ONLY),
    ("Aerial: Blend weight", AtmosphereDiagnosticView.AERIAL_BLEND_WEIGHT),
    (
        "Atmo: Aerial transmittance",
        AtmosphereDiagnosticView.AERIAL_TRANSMITTANCE,
    ),
    ("Debug: Surface mask", AtmosphereDiagnosticView.SURFACE_MASK),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi M0 解析行星渲染器")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "planet.json")
    parser.add_argument(
        "--backend", choices=("auto", "cuda", "vulkan", "cpu"), default="auto"
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-hdr", type=Path)
    parser.add_argument("--benchmark-frames", type=int, default=20)
    parser.add_argument(
        "--altitude-m", type=float, help="覆盖初始径向高度，用于离线检查"
    )
    parser.add_argument("--yaw-degrees", type=float, help="覆盖初始 yaw")
    parser.add_argument("--pitch-degrees", type=float, help="覆盖初始 pitch")
    return parser.parse_args(argv)


def initialize_taichi(backend: str) -> None:
    arch = {"auto": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[
        backend
    ]
    ti.init(arch=arch, default_fp=ti.f32, offline_cache=True)


def initial_state(config: M0Config):
    planet = PlanetModel(config.planet_radius_m)
    position = planet.surface_position(
        np.array([0.0, 0.0, 1.0]), config.initial_altitude_m
    )
    camera = PlanetCamera(
        position, config.yaw_degrees, config.pitch_degrees, config.vertical_fov_degrees
    )
    frame = planet.local_frame(position)
    azimuth, elevation = math.radians(config.sun_azimuth_degrees), math.radians(
        config.sun_elevation_degrees
    )
    sun_local = np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ]
    )
    lighting = LightingState(
        frame.local_to_global_direction(sun_local),
        config.sun_angular_radius_degrees,
        config.solar_irradiance,
    )
    origin = FloatingOrigin(position.copy(), config.floating_origin_threshold_m)
    return planet, camera, StaticLightingProvider(lighting), origin


def _active_surface_height_m(
    terrain: CubeSphereTerrain,
    direction: np.ndarray,
    ocean_enabled: bool,
) -> float:
    """Return the opaque surface height used for camera clearance.

    Terrain below the reference radius is seabed when the sea-level coverage
    pass is enabled.  The camera must therefore be constrained against height
    zero rather than against the visually hidden seabed.
    """

    terrain_height = terrain.describe_surface(direction).height_m
    return opaque_surface_height_m(terrain_height, ocean_enabled)


def _constrain_camera_to_surface(
    planet: PlanetModel,
    camera: PlanetCamera,
    terrain: CubeSphereTerrain,
    ocean_enabled: bool,
    minimum_clearance_m: float = 0.5,
) -> float:
    radius = float(np.linalg.norm(camera.position_global))
    direction = camera.position_global / max(radius, 1.0)
    surface_height = _active_surface_height_m(
        terrain,
        direction,
        ocean_enabled,
    )
    minimum_radius = planet.radius_m + surface_height + minimum_clearance_m
    if radius < minimum_radius:
        camera.position_global = direction * minimum_radius
    return surface_height


def save_image(
    renderer: PlanetRenderer, path: Path | None, hdr_path: Path | None
) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ti.tools.imwrite(renderer.display_numpy(), str(path))
        print(f"已保存图像：{path.resolve()}")
    if hdr_path is not None:
        hdr_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hdr_path, renderer.hdr_numpy())
        print(f"已保存线性 HDR：{hdr_path.resolve()}")


def run_preview(
    renderer: PlanetRenderer,
    config: M0Config,
    planet: PlanetModel,
    camera: PlanetCamera,
    provider: StaticLightingProvider,
    origin: FloatingOrigin,
    terrain: CubeSphereTerrain,
) -> None:
    window = ti.ui.Window(
        "Taichi Planet Renderer - M3.4", (config.width, config.height), vsync=True
    )
    canvas, gui = window.get_canvas(), window.get_gui()
    previous_mouse: tuple[float, float] | None = None
    mouse_sensitivity, speed_exponent = 180.0, 2.0
    exposure = config.postprocess.exposure_ev
    last_time = time.perf_counter()
    preview_start_time = last_time
    preview_frame = 0
    camera_input_frozen = False
    terrain_lod_frozen = False
    terrain_future: Future | None = None
    terrain_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="terrain-stream",
    )
    terrain_stats = terrain.tile_manager.stats
    active_surface_height = _constrain_camera_to_surface(
        planet,
        camera,
        terrain,
        config.ocean.surface_enabled,
    )
    while window.running:
        now = time.perf_counter()
        dt, last_time = min(now - last_time, 0.1), now
        if window.is_pressed(ti.ui.ESCAPE):
            break
        cursor = window.get_cursor_pos()
        over_panel = cursor[0] >= 0.70
        if (
            not camera_input_frozen
            and window.is_pressed(ti.ui.LMB)
            and not over_panel
        ):
            if previous_mouse is not None:
                camera.yaw_degrees += (
                    cursor[0] - previous_mouse[0]
                ) * mouse_sensitivity
                camera.pitch_degrees = float(
                    np.clip(
                        camera.pitch_degrees
                        + (cursor[1] - previous_mouse[1]) * mouse_sensitivity,
                        -89.0,
                        89.0,
                    )
                )
            previous_mouse = cursor
        else:
            previous_mouse = None

        if not camera_input_frozen:
            yaw = math.radians(camera.yaw_degrees)
            forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
            motion = forward * (
                int(window.is_pressed("w")) - int(window.is_pressed("s"))
            )
            motion += right * (
                int(window.is_pressed("d")) - int(window.is_pressed("a"))
            )
            motion[1] += int(window.is_pressed(ti.ui.SPACE)) - int(
                window.is_pressed(ti.ui.SHIFT)
            )
            length = float(np.linalg.norm(motion))
            if length > 0.0:
                distance = 10.0**speed_exponent * dt
                camera.move_local(planet, *(motion / length * distance))
                active_surface_height = _constrain_camera_to_surface(
                    planet,
                    camera,
                    terrain,
                    config.ocean.surface_enabled,
                )
        origin.update(camera.position_global)

        if terrain_future is not None and terrain_future.done():
            terrain_frame = terrain_future.result()
            renderer.apply_terrain_frame(terrain_frame)
            terrain_stats = terrain_frame.stats
            terrain_future = None
        if (
            terrain_future is None
            and not terrain_lod_frozen
            and preview_frame % config.terrain_update_interval_frames == 0
        ):
            camera_snapshot = PlanetCamera(
                camera.position_global.copy(),
                camera.yaw_degrees,
                camera.pitch_degrees,
                camera.vertical_fov_degrees,
            )
            terrain_future = terrain_executor.submit(
                terrain.update,
                camera_snapshot,
                config.width,
                config.height,
            )
        renderer.render(
            planet,
            camera,
            provider.snapshot(),
            config.surface_albedo,
            exposure,
            now - preview_start_time,
        )
        canvas.set_image(renderer.display)
        altitude = planet.altitude_m(camera.position_global)
        with gui.sub_window("M0 Planet", 0.705, 0.02, 0.28, 0.96) as panel:
            panel.text(f"Altitude: {altitude:,.2f} m")
            panel.text(
                f"Surface clearance: {altitude - active_surface_height:,.2f} m"
            )
            panel.text(
                f"Horizon: {planet.horizon_distance_m(altitude) / 1000.0:,.2f} km"
            )
            panel.text(f"Origin revision: {origin.revision}")
            stats = terrain_stats
            panel.text(f"LOD: {stats.min_lod}..{stats.max_lod}")
            panel.text(
                f"Patch D/R/V: {stats.desired_patches}/{stats.resident_patches}/{stats.render_patches}"
            )
            panel.text(
                f"GPU slots: {stats.resident_patches}/"
                f"{config.terrain_max_gpu_patches}"
            )
            panel.text(
                f"State requested/ready: {stats.requested_patches}/{stats.ready_patches}"
            )
            panel.text(
                f"Select/build/upload: {stats.selection_ms:.2f}/{stats.build_ms:.2f}/{stats.upload_ms:.2f} ms"
            )
            panel.text(f"Cache hit: {stats.cache_hit_rate*100:.1f}%")
            panel.text(
                f"Terrain update: 1/{config.terrain_update_interval_frames} frames"
            )
            panel.text(
                "Terrain worker: "
                + ("busy" if terrain_future is not None else "idle")
            )
            panel.text(
                (
                    "Surface: material"
                    if renderer.debug_view == 0
                    else (
                        "Surface: height"
                        if renderer.debug_view == 1
                        else (
                            "Surface: LOD"
                            if renderer.debug_view == 2
                            else "Surface: patch ID"
                        )
                    )
                )
            )
            panel.text(
                f"Raster: {renderer.vertex_count:,} vertices / {renderer.triangle_count:,} triangles"
            )
            panel.text(
                "Tile max/overflow: "
                f"{renderer.last_max_tile_candidates}/"
                f"{renderer.last_tile_overflow}"
            )
            panel.text(
                "Atmosphere LUT T/M/S/A: "
                f"{renderer.atmosphere_renderer.transmittance_rebuilds}/"
                f"{renderer.atmosphere_renderer.multi_scattering_rebuilds}/"
                f"{renderer.atmosphere_renderer.sky_view_rebuilds}/"
                f"{renderer.atmosphere_renderer.aerial_rebuilds}"
            )
            panel.text(
                "Atmosphere view: "
                f"{renderer.atmosphere_diagnostic_view.label}"
            )
            lighting_state = provider.snapshot()
            panel.text(
                "Sun E / disk L: "
                f"{max(lighting_state.solar_irradiance):.3g} / "
                f"{max(lighting_state.sun_disk_radiance):.3g}"
            )
            panel.text(
                "Bloom: "
                f"{config.postprocess.bloom_strength:.3f} strength / "
                f"{config.postprocess.bloom_passes} passes / "
                f"{config.postprocess.bloom_clamp:.1f} clamp"
            )
            panel.text(
                "Atmosphere LUTs: "
                + (
                    "frozen"
                    if renderer.atmosphere_renderer.luts_frozen
                    else "live"
                )
            )
            sky_altitude = renderer.atmosphere_renderer.sky_snapshot_altitude_m
            if sky_altitude is not None:
                panel.text(f"Sky snapshot altitude: {sky_altitude:,.2f} m")
            sky_sun_cosine = (
                renderer.atmosphere_renderer.sky_snapshot_sun_cosine
            )
            if sky_sun_cosine is not None:
                panel.text(f"Sky snapshot sun cosine: {sky_sun_cosine:.6f}")
            panel.text(
                "Terrain LOD: "
                + ("frozen" if terrain_lod_frozen else "live")
                + (" (finishing request)" if terrain_future is not None else "")
            )
            panel.text(
                "Camera input: "
                + ("frozen" if camera_input_frozen else "live")
            )
            relative = origin.relative_f32(camera.position_global)
            panel.text(
                f"Camera relative: {relative[0]:.1f}, {relative[1]:.1f}, {relative[2]:.1f}"
            )
            global_position = camera.position_global
            panel.text(
                "Camera global: "
                f"{global_position[0]:.1f}, {global_position[1]:.1f}, "
                f"{global_position[2]:.1f} m"
            )
            panel.text(
                f"Yaw/Pitch: {camera.yaw_degrees:.3f}/{camera.pitch_degrees:.3f} deg"
            )
            speed_exponent = panel.slider_float(
                "Speed log10(m/s)", speed_exponent, 0.0, 7.0
            )
            mouse_sensitivity = panel.slider_float(
                "Mouse sensitivity", mouse_sensitivity, 30.0, 500.0
            )
            camera.vertical_fov_degrees = panel.slider_float(
                "Vertical FOV", camera.vertical_fov_degrees, 20.0, 120.0
            )
            exposure = panel.slider_float("Exposure EV", exposure, -5.0, 5.0)
            if panel.button("Surface: Material"):
                renderer.debug_view = 0
            if panel.button("Debug: Height"):
                renderer.debug_view = 1
            if panel.button("Debug: LOD"):
                renderer.debug_view = 2
            if panel.button("Debug: Patch ID"):
                renderer.debug_view = 3
            for label, diagnostic_view in _ATMOSPHERE_DIAGNOSTIC_BUTTONS:
                if panel.button(label):
                    renderer.atmosphere_diagnostic_view = diagnostic_view
            if panel.button(
                "Resume atmosphere LUTs"
                if renderer.atmosphere_renderer.luts_frozen
                else "Freeze atmosphere LUTs"
            ):
                renderer.atmosphere_renderer.set_luts_frozen(
                    not renderer.atmosphere_renderer.luts_frozen
                )
            if panel.button(
                "Resume terrain LOD"
                if terrain_lod_frozen
                else "Freeze terrain LOD"
            ):
                terrain_lod_frozen = not terrain_lod_frozen
            if panel.button(
                "Resume camera input"
                if camera_input_frozen
                else "Freeze camera input"
            ):
                camera_input_frozen = not camera_input_frozen
            if panel.button("Surface (2 m)"):
                direction = camera.position_global / np.linalg.norm(
                    camera.position_global
                )
                active_surface_height = _active_surface_height_m(
                    terrain,
                    direction,
                    config.ocean.surface_enabled,
                )
                camera.position_global = planet.surface_position(
                    direction, active_surface_height + 2.0
                )
            if panel.button("High atmosphere (50 km)"):
                camera.position_global = planet.surface_position(
                    camera.position_global, 50_000.0
                )
            if panel.button("Space (2,000 km)"):
                camera.position_global = planet.surface_position(
                    camera.position_global, 2_000_000.0
                )
            panel.text("WASD | Space/Shift | hold LMB")
        window.show()
        preview_frame += 1
    terrain_executor.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        initialize_taichi(args.backend)
        planet, camera, provider, origin = initial_state(config)
        if args.altitude_m is not None:
            camera.position_global = planet.surface_position(
                camera.position_global, max(args.altitude_m, 0.5)
            )
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
            create_terrain_model(config.terrain),
            settings,
        )
        requested_altitude = planet.altitude_m(camera.position_global)
        if requested_altitude < 20_000.0:
            direction = camera.position_global / np.linalg.norm(camera.position_global)
            surface_height = _active_surface_height_m(
                terrain,
                direction,
                config.ocean.surface_enabled,
            )
            camera.position_global = planet.surface_position(
                direction, surface_height + max(requested_altitude, 2.0)
            )
            origin.origin_global = camera.position_global.copy()
        renderer = PlanetRenderer(
            config.width,
            config.height,
            terrain.height_model,
            config.atmosphere,
            config.postprocess,
            config.terrain_max_gpu_patches,
            config.terrain_patch_resolution,
            config.ocean,
            config.space,
        )
        # 两个固定预算 bootstrap tick 使六个根 patch 可作为初始 fallback；不等待细分完成。
        renderer.apply_terrain_frame(
            terrain.update(camera, config.width, config.height)
        )
        renderer.apply_terrain_frame(
            terrain.update(camera, config.width, config.height)
        )
        renderer.warmup_raster_paths(
            planet,
            camera,
            provider.snapshot(),
            config.surface_albedo,
            config.postprocess.exposure_ev,
        )
        render_call = lambda: renderer.render(
            planet,
            camera,
            provider.snapshot(),
            config.surface_albedo,
            config.postprocess.exposure_ev,
        )
        timing = renderer.benchmark(render_call, args.benchmark_frames)
        raster_stats = renderer.stats()
        print(f"后端：{ti.lang.impl.current_cfg().arch}")
        print(f"首次渲染（含 JIT）：{timing.jit_seconds * 1000:.2f} ms")
        print(
            f"稳定帧时间：{timing.average_seconds * 1000:.2f} ms ({timing.frames_per_second:.1f} FPS)"
        )
        print(f"相机高度：{planet.altitude_m(camera.position_global):.3f} m")
        print(
            f"解析地平线距离：{planet.horizon_distance_m(planet.altitude_m(camera.position_global)):.3f} m"
        )
        print(
            f"光栅统计：{raster_stats.patches} 分块，{raster_stats.vertices} 顶点，{raster_stats.triangles} 三角形，tile 溢出 {raster_stats.tile_overflow}"
        )
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
