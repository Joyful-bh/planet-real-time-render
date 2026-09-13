"""参考大气原型命令行入口；不属于默认稳定渲染路径。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

from .config import load_config
from .renderer import SkyRenderer


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi 程序化天空渲染器")
    parser.add_argument("--preset", choices=("day", "dawn", "sunset"), default="day", help="内置天空预设")
    parser.add_argument("--config", type=Path, help="自定义 JSON 配置；指定后覆盖 --preset")
    parser.add_argument("--backend", choices=("auto", "cuda", "vulkan", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="离线输出 PNG 路径")
    parser.add_argument("--save-hdr", type=Path, help="可选：保存线性 HDR NumPy 数组 (.npy)")
    parser.add_argument("--preview", action="store_true", help="打开交互预览；方向键旋转相机，Esc 退出")
    parser.add_argument("--benchmark-frames", type=int, default=20, help="预热后的计时帧数")
    return parser.parse_args(argv)


def initialize_taichi(backend: str) -> None:
    architectures = {"auto": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}
    # 禁用跨进程离线缓存，避免共享缓存锁影响独立运行；进程内 JIT 仍会复用。
    ti.init(arch=architectures[backend], default_fp=ti.f32, offline_cache=False)


def save_outputs(renderer: SkyRenderer, output: Path | None, hdr_output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        ti.tools.imwrite(renderer.display_numpy(), str(output))
        print(f"已保存显示图像：{output.resolve()}")
    if hdr_output is not None:
        hdr_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(hdr_output, renderer.hdr_numpy())
        print(f"已保存线性 HDR：{hdr_output.resolve()}")


def run_preview(renderer: SkyRenderer) -> None:
    c = renderer.config
    window = ti.ui.Window("Taichi Atmosphere", (c.width, c.height), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    yaw, pitch = c.yaw_degrees, c.pitch_degrees
    position = np.array([0.0, c.camera_altitude_km, 0.0], dtype=np.float64)
    move_speed = 0.25
    mouse_sensitivity = 180.0
    previous_mouse: tuple[float, float] | None = None
    last_time = time.perf_counter()
    params: dict[str, object] = {
        "sun_azimuth_degrees": c.sun_azimuth_degrees,
        "sun_elevation_degrees": c.sun_elevation_degrees,
        "vertical_fov_degrees": c.vertical_fov_degrees,
        "atmosphere_height_km": c.atmosphere_height_km,
        "rayleigh_scale_height_km": c.rayleigh_scale_height_km,
        "mie_scale_height_km": c.mie_scale_height_km,
        "mie_multiplier": 1.0,
        "mie_g": c.mie_g,
        "ground_albedo": c.ground_albedo,
        "exposure_ev": c.exposure_ev,
        "use_aces": c.tone_mapper == "aces",
    }
    while window.running:
        now = time.perf_counter()
        dt = min(now - last_time, 0.1)
        last_time = now
        if window.is_pressed(ti.ui.ESCAPE):
            break

        mouse = window.get_cursor_pos()
        over_panel = mouse[0] >= 0.69 and mouse[1] >= 0.02
        if window.is_pressed(ti.ui.LMB) and not over_panel:
            if previous_mouse is not None:
                yaw += (mouse[0] - previous_mouse[0]) * mouse_sensitivity
                pitch += (mouse[1] - previous_mouse[1]) * mouse_sensitivity
                pitch = min(max(pitch, -89.0), 89.0)
            previous_mouse = mouse
        else:
            previous_mouse = None

        yaw_radians = math.radians(yaw)
        forward = np.array([math.sin(yaw_radians), 0.0, math.cos(yaw_radians)])
        right = np.array([math.cos(yaw_radians), 0.0, -math.sin(yaw_radians)])
        movement = np.zeros(3)
        movement += forward * (int(window.is_pressed("w")) - int(window.is_pressed("s")))
        movement += right * (int(window.is_pressed("d")) - int(window.is_pressed("a")))
        movement[1] += int(window.is_pressed(ti.ui.SPACE)) - int(window.is_pressed(ti.ui.SHIFT))
        norm = np.linalg.norm(movement)
        if norm > 0.0:
            position += movement / norm * move_speed * dt
            position[1] = max(position[1], 0.0001)

        with gui.sub_window("Atmosphere Parameters", 0.70, 0.02, 0.285, 0.94) as panel:
            panel.text(f"Position km: {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}")
            panel.text(f"View: yaw {yaw:.1f}, pitch {pitch:.1f}")
            move_speed = panel.slider_float("Move speed (km/s)", move_speed, 0.01, 10.0)
            mouse_sensitivity = panel.slider_float("Mouse sensitivity", mouse_sensitivity, 30.0, 500.0)
            params["vertical_fov_degrees"] = panel.slider_float("Vertical FOV", float(params["vertical_fov_degrees"]), 20.0, 120.0)
            params["sun_azimuth_degrees"] = panel.slider_float("Sun azimuth", float(params["sun_azimuth_degrees"]), -180.0, 180.0)
            params["sun_elevation_degrees"] = panel.slider_float("Sun elevation", float(params["sun_elevation_degrees"]), -12.0, 90.0)
            params["atmosphere_height_km"] = panel.slider_float("Atmosphere height km", float(params["atmosphere_height_km"]), 20.0, 200.0)
            params["rayleigh_scale_height_km"] = panel.slider_float("Rayleigh scale km", float(params["rayleigh_scale_height_km"]), 4.0, 16.0)
            params["mie_scale_height_km"] = panel.slider_float("Mie scale km", float(params["mie_scale_height_km"]), 0.2, 5.0)
            params["mie_multiplier"] = panel.slider_float("Mie density", float(params["mie_multiplier"]), 0.0, 5.0)
            params["mie_g"] = panel.slider_float("Mie anisotropy g", float(params["mie_g"]), 0.0, 0.95)
            params["exposure_ev"] = panel.slider_float("Exposure EV", float(params["exposure_ev"]), -5.0, 5.0)
            params["ground_albedo"] = panel.color_edit_3("Ground albedo", params["ground_albedo"])
            params["use_aces"] = panel.checkbox("ACES (off: Reinhard)", bool(params["use_aces"]))
            if panel.button("Reset camera"):
                position[:] = (0.0, c.camera_altitude_km, 0.0)
                yaw, pitch = c.yaw_degrees, c.pitch_degrees
            panel.text("WASD move | Space up | Shift down")
            panel.text("Hold LMB and drag to look")

        renderer.render(yaw, pitch, tuple(position), params)
        canvas.set_image(renderer.display)
        window.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config or ROOT / "configs" / f"{args.preset}.json"
    try:
        config = load_config(config_path)
        initialize_taichi(args.backend)
        renderer = SkyRenderer(config)
        timing = renderer.benchmark(args.benchmark_frames)
        print(f"后端：{ti.lang.impl.current_cfg().arch}")
        print(f"首次渲染（含 JIT）：{timing.jit_seconds * 1000:.2f} ms")
        print(f"稳定帧时间：{timing.average_seconds * 1000:.2f} ms ({timing.frames_per_second:.1f} FPS)")
        save_outputs(renderer, args.output, args.save_hdr)
        if args.preview:
            run_preview(renderer)
        if args.output is None and not args.preview:
            default_output = ROOT / "output" / f"{args.preset}.png"
            save_outputs(renderer, default_output, None)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
