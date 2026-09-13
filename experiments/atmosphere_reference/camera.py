"""旧原型相机射线生成，仅用于参考对照。"""

import taichi as ti


@ti.func
def generate_ray(
    pixel: ti.template(),
    resolution: ti.template(),
    yaw_radians: ti.f32,
    pitch_radians: ti.f32,
    vertical_fov_radians: ti.f32,
):
    """返回给定像素中心的世界空间单位观察方向，角度单位为弧度。"""
    uv = (ti.cast(pixel, ti.f32) + 0.5) / ti.cast(resolution, ti.f32)
    aspect = ti.cast(resolution[0], ti.f32) / ti.cast(resolution[1], ti.f32)
    screen = ti.Vector([(uv.x * 2.0 - 1.0) * aspect, uv.y * 2.0 - 1.0])
    focal = 1.0 / ti.tan(vertical_fov_radians * 0.5)
    local = ti.Vector([screen.x, screen.y, focal]).normalized()

    cy, sy = ti.cos(yaw_radians), ti.sin(yaw_radians)
    cp, sp = ti.cos(pitch_radians), ti.sin(pitch_radians)
    forward = ti.Vector([sy * cp, sp, cy * cp])
    right = ti.Vector([cy, 0.0, -sy])
    up = forward.cross(right)
    return (right * local.x + up * local.y + forward * local.z).normalized()
