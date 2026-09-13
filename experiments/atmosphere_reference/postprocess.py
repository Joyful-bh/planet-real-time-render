"""参考原型显示变换。"""

import taichi as ti


@ti.func
def aces_fitted(color: ti.template()):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ti.math.clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0)


@ti.func
def linear_to_srgb(color: ti.template()):
    lo = color * 12.92
    hi = 1.055 * ti.pow(ti.max(color, 0.0), 1.0 / 2.4) - 0.055
    return ti.Vector([lo[i] if color[i] <= 0.0031308 else hi[i] for i in ti.static(range(3))])


@ti.func
def display_transform(color: ti.template(), exposure_ev: ti.f32, use_aces: ti.i32):
    exposed = color * ti.pow(2.0, exposure_ev)
    mapped = exposed
    if use_aces != 0:
        mapped = aces_fitted(exposed)
    else:
        mapped = mapped / (1.0 + mapped)
    return linear_to_srgb(ti.math.clamp(mapped, 0.0, 1.0))
