"""线性 HDR 到显示颜色的 Taichi 变换。"""

import taichi as ti


@ti.func
def display_transform(color: ti.template(), exposure_ev: ti.f32):
    value = color * ti.pow(2.0, exposure_ev)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = ti.math.clamp(
        (value * (a * value + b)) / (value * (c * value + d) + e), 0.0, 1.0
    )
    lo = mapped * 12.92
    hi = 1.055 * ti.pow(mapped, 1.0 / 2.4) - 0.055
    return ti.Vector(
        [lo[i] if mapped[i] <= 0.0031308 else hi[i] for i in ti.static(range(3))]
    )
