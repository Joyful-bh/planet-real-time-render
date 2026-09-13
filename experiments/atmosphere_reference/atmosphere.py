"""球形行星大气的 Rayleigh/Mie 单次散射参考实现；距离单位为千米。"""

import taichi as ti


@ti.func
def sphere_near(o: ti.template(), d: ti.template(), radius: ti.f32) -> ti.f32:
    b, q = o.dot(d), -1.0
    det = b * b - o.dot(o) + radius * radius
    if det >= 0.0:
        q = -b - ti.sqrt(det)
    return q


@ti.func
def sphere_far(o: ti.template(), d: ti.template(), radius: ti.f32) -> ti.f32:
    b, q = o.dot(d), -1.0
    det = b * b - o.dot(o) + radius * radius
    if det >= 0.0:
        q = -b + ti.sqrt(det)
    return q


@ti.func
def densities(p: ti.template(), radius: ti.f32, hr: ti.f32, hm: ti.f32):
    altitude = ti.max(p.norm() - radius, 0.0)
    return ti.Vector([ti.exp(-altitude / hr), ti.exp(-altitude / hm)])


@ti.func
def sun_depth(p: ti.template(), sun: ti.template(), radius: ti.f32, top: ti.f32, hr: ti.f32, hm: ti.f32):
    """返回 Rayleigh/Mie 柱密度和行星遮挡标记。"""
    result = ti.Vector([0.0, 0.0, 0.0])
    if sphere_near(p, sun, radius) > 1.0e-4:
        result.z = 1.0
    else:
        length = ti.max(sphere_far(p, sun, top), 0.0)
        ds = length / 8.0
        for j in range(8):
            result.xy += densities(p + sun * ((ti.cast(j, ti.f32) + 0.5) * ds), radius, hr, hm) * ds
    return result


@ti.func
def attenuation(depth: ti.template(), beta_r: ti.template(), beta_m_ext: ti.template()):
    return ti.exp(-ti.min(beta_r * depth.x + beta_m_ext * depth.y, 80.0))


@ti.func
def rayleigh_phase(mu: ti.f32) -> ti.f32:
    return 3.0 * (1.0 + mu * mu) / (16.0 * ti.math.pi)


@ti.func
def mie_phase(mu: ti.f32, g: ti.f32) -> ti.f32:
    denom = ti.pow(ti.max(1.0 + g * g - 2.0 * g * mu, 1.0e-4), 1.5)
    return (1.0 - g * g) / (4.0 * ti.math.pi * denom)


@ti.func
def shade_atmosphere(
    ray: ti.template(), camera: ti.template(), sun: ti.template(), radius: ti.f32, top: ti.f32,
    hr: ti.f32, hm: ti.f32, beta_r: ti.template(), beta_m_sca: ti.template(),
    beta_m_ext: ti.template(), g: ti.f32, solar: ti.template(), sun_radiance: ti.template(),
    sun_radius: ti.f32, ground_albedo: ti.template(),
):
    """积分观察射线上的物理单次散射，并合成局部平面地面与太阳圆盘。"""
    end = sphere_far(camera, ray, top)
    plane_hit = -1.0
    # 渲染地面是初始观测点处的局部切平面 y=planet_radius；
    # 大气密度仍严格由到球心的径向高度决定。
    camera_height = camera.y - radius
    if ray.y < -1.0e-6:
        plane_hit = camera_height / -ray.y
        end = ti.min(end, plane_hit)
    ds = ti.max(end, 0.0) / 32.0
    view_depth = ti.Vector([0.0, 0.0])
    sum_r, sum_m = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
    for i in range(32):
        p = camera + ray * ((ti.cast(i, ti.f32) + 0.5) * ds)
        rho = densities(p, radius, hr, hm)
        light_depth = sun_depth(p, sun, radius, top, hr, hm)
        if light_depth.z < 0.5:
            trans = attenuation(view_depth + light_depth.xy, beta_r, beta_m_ext)
            sum_r += trans * rho.x * ds
            sum_m += trans * rho.y * ds
        view_depth += rho * ds
    mu = ti.math.clamp(ray.dot(sun), -1.0, 1.0)
    radiance = solar * (beta_r * sum_r * rayleigh_phase(mu) + beta_m_sca * sum_m * mie_phase(mu, g))
    view_trans = attenuation(view_depth, beta_r, beta_m_ext)
    if plane_hit > 0.0:
        light_depth = sun_depth(camera + ray * plane_hit, sun, radius, top, hr, hm)
        if light_depth.z < 0.5:
            direct = solar * attenuation(light_depth.xy, beta_r, beta_m_ext) * ti.max(sun.y, 0.0) / ti.math.pi
            radiance += ground_albedo * direct * view_trans
    if mu >= ti.cos(sun_radius):
        direct_depth = sun_depth(camera, sun, radius, top, hr, hm)
        if direct_depth.z < 0.5:
            radiance += sun_radiance * attenuation(direct_depth.xy, beta_r, beta_m_ext)
    return ti.max(radiance, 0.0)
