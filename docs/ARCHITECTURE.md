# 行星实时渲染器架构与路线

## 1. 架构原则

项目只有一个连续的行星世界。地表视角和太空视角不是两个独立渲染模式；LOD、采样预算和后处理可以随高度变化，但坐标、太阳、大气和地表数据必须连续。

系统优先建立可以运行、观察和测量的垂直切片。任何里程碑都应包含交互飞行、实际画面、数值检查和性能数据。

## 2. 坐标与精度

- 行星中心是全局坐标原点，长度基准单位为米。
- CPU 保存 `float64` 相机位置和行星级计算。
- GPU 顶点、射线和局部对象使用以当前相机为原点的 `float32` 坐标。
- 每帧从双精度全局坐标构造局部东—北—天或等价正交基。
- 高度定义为相机到行星中心距离减参考半径，不能直接使用世界 Y。
- 浮动原点变化不得造成相机姿态、地形分块、云或太阳方向跳变。
- 在极区、立方体面边缘、大气层顶和地表以下位置设置明确数值保护。

## 3. 子系统与数据流

### `planet`

定义行星半径、海平面、大气层顶、全局/局部坐标转换、经纬度和切线基。它不生成地形，也不计算散射。

### `camera`

保存双精度行星位置、姿态、速度和投影参数，负责输入驱动及观察射线。它不决定地形 LOD。

### `time`

维护并输出不可变 `TimeState`。其中 `runtime_seconds/frame_index` 单调递增，供缓存、时序重建和诊断使用；`world_time_seconds/delta_world_seconds` 可暂停、缩放和显式跳转，供天体、天气与动画使用。它不计算太阳、天气或动画结果。

### `celestial`

在 M5 开始消费 `TimeState` 与行星自转参数，成为 `LightingState` 的动态生产者，替换 M0 的静态生产者。任一时刻只能有一个激活的照明状态生产者。当前只需解析昼夜周期，不实现轨道力学；季节和轴倾角可在不改变消费者接口的前提下后续加入。

### `terrain`

负责球面分块、四叉树 LOD、可见性和高度源调度，并输出网格生成所需的轻量 descriptor。地形配置分为两层：

- 外层 `TerrainConfig` 只包含稳定的生成器 ID 和不透明 `params`；
- `TerrainFactory` 通过显式 Registry 找到生成器，并将 `params` 转换为该算法自己的强类型配置；
- `procedural_landforms_v1` 是默认分区地貌生成器，`procedural_fbm_v1` 保留为较小的参考实现；两者拥有各自的强类型内部配置，未来算法同样不得向外层信封增加地貌语义字段；
- `DemTerrainModel` 预留真实数据切片、重投影、缓存和缺失值处理边界。

高度源只返回规范化位置对应的高度及必要元数据，不负责渲染。

### `surface`

定义地形着色所需的表面语义，并向未来覆盖物提供法线、坡度、曲率、材质/生物群系权重和确定性 seed。覆盖物身份使用与渲染 LOD 无关的规范球面 `SurfaceCellId`，不能直接使用包含 level 的地形 patch key；M2 同时定义分块可见/驻留事件和 descriptor 生命周期。当前不生成植被或其他覆盖物实例。

### `ocean`

负责海平面球面、波浪法线、吸收、反射和海岸混合。水体遮挡关系必须与地形高度一致。

### `atmosphere`

负责密度、消光、散射、LUT 和空气透视。直接单次散射积分是验证基准；实时默认路径逐步迁移到 LUT。不得包含云密度或地形材质。

### `clouds`

负责把只读 `WeatherState` 解释为有限云壳中的密度、光照、自阴影和体积结果。它不拥有天气图、风场或天气演化，也不向大气和海洋反向输出天气参数。

### `weather`

产生只读 `WeatherState`，描述球面云量、风、湿度、降水潜势和气溶胶修正。早期可以是静态预设，后期可以演化或由外部数据驱动；它不包含云 Ray Marching、海洋 BRDF 或大气散射实现。

### `space`

负责星空方向数据和太空背景。它不自行判断昼夜；可见性由大气透射、曝光和遮挡共同决定。

### `lighting`

定义 `LightingState` 数据契约并向地形、海洋、大气和云提供一致输入。M0 提供静态生产者，M5 由 `celestial(TimeState)` 动态生产者替换；`lighting` 本身不维护第二份太阳状态。

### `renderer`

负责资源生命周期、kernel 调度、可见分块上传、渲染目标、合成、分辨率策略和统计，不包含具体地形噪声或散射公式。

### `postprocess` 与 `io`

前者负责曝光、Bloom、色调映射和显示编码；后者负责配置、图像、DEM、缓存和资产元数据。

跨模块状态采用单向快照：`TimeState → Celestial/Weather → LightingState/WeatherState → render consumers`。M0 建立静态 `LightingState`，M3 建立静态默认 `WeatherState`；后续里程碑只替换对应生产者。消费者不得修改生产者，也不得各自维护重复的太阳、时间或风状态。

## 4. 地形表示决策

首选立方体球四叉树：六个立方体面映射到球面，每面递归划分规则分块。原因是拓扑规则、适合 GPU 网格生成、容易进行面内四叉树 LOD，并避免单一经纬网格的极点退化。

每个分块由键 `(face, level, x, y)` 唯一标识。LOD 使用投影到屏幕的几何误差决定；同时进行视锥裁剪和基于球体的地平线裁剪。相邻分块层级差限制为一级，并通过裙边、边界重采样或拓扑拼接消除裂缝。具体方案在原型测量后确定，不提前固化。

## 5. 大气正确性

大气 API 在语义上接受行星位置和方向，但实现必须遵守精度分层：CPU 参考计算与 LUT 生成可以使用 `float64` 全局坐标；GPU 查询只能接收相机相对 `float32` 位置以及安全重基化后的行星中心与半径，或经过验证的高低位拆分，不能逐像素传递行星尺度绝对 `float32` 坐标。至少覆盖四类观察条件：

1. 地表向上看天空；
2. 地表沿地平线观察；
3. 大气内向下看地表；
4. 大气外观察行星边缘与背光面。

验证基准包含球壳求交、指数密度、Rayleigh/Mie 消光与单次散射。实时 LUT 必须明确维度、参数化、颜色空间、太阳角度依赖和重建方法，并与参考积分器在固定采样点比较误差。

## 6. 分阶段里程碑

### M0：重新打稳行星基础

状态：已完成。

- 新的双精度行星相机和浮动原点；
- 球体/立方体球线框与纯色表面；
- 从地表连续飞到太空；
- 正确的球形地平线和太阳方向；
- 定义 `LightingState` 并由唯一的静态照明生产者提供；
- 地表、轨道高度和大气层外的精度测试。

验收：连续飞行无明显抖动、裂开或坐标跳变；相机高度和地平线距离符合解析值。

### M1：自定义表面渲染基础

状态：已完成。当前使用均匀 48×48 六面立方体球验证管线；近地表轮廓的低细分误差由 M2 分块 LOD 解决，不在 M1 复制临时细节系统。

- Taichi 顶点变换、三角形分桶与光栅化；
- 深度缓冲和基础 G-buffer；
- 立方体球网格与法线调试视图；
- 明确背景、体积区间和最近表面命中的合成契约。

验收：球体在地表与太空视角稳定光栅化，深度可被后续大气和海洋消费。

### M2：程序生成球面地形

状态：已完成增量 Mixed-LOD 闭环。`terrain_lod` 维护持久叶集合，以 patch SSE 和独立 split/merge 阈值渐进改变拓扑，并把相邻层级差限制为一；`terrain_streaming` 维护 Desired/Resident/Render 三集合、生命周期、父级 fallback、请求优先级、预算和 LRU slot；`height` 定义程序/DEM provider 契约；`terrain_renderer` 持有 GPU 几何 slot，并在 Taichi 中生成程序高度、相机相对顶点和法线；`renderer` 仅消费这些字段并执行变换、光栅化和合成。Render Set 单独做视锥与地平线裁剪。第一版只在连接粗邻居或缺失邻居的细边启用 skirt；后续用 edge stitching 和 geomorph 替换。

- 六面分块和基础四叉树 LOD；
- 确定性低频大陆与多尺度山地；
- 法线、基础岩土材质和 LOD 裂缝处理；
- 视锥与地平线裁剪、分块统计面板；
- 输出稳定 `SurfaceDescriptor`，但不实现植被等覆盖物。
- 定义与渲染 LOD 无关的 `SurfaceCellId`、分块驻留事件和 descriptor 生命周期。

验收：地表与太空视角均能辨认同一地形；跨面和跨 LOD 无明显裂缝。

### M3：全高度物理大气

- 以当前参考积分器验证数学结果，为行星相机建立新的大气查询接口；
- 加入臭氧吸收和行星表面遮挡；
- 实现并验证 Transmittance、Multi-scattering、Sky-view、Aerial-perspective LUT；
- 大气应用于天空和地形，并提供后续海洋、云与星空可调用的接口。
- 建立正式但静态默认的 `WeatherState`，大气从中读取气溶胶倍率。

验收：四类观察条件连续；天空与地形的地平线、大气边缘和昼夜分界无模式切换。

### M4：球面海洋与水面光照

- 海平面球面与地形海岸判定；
- 采用共享 `LightingState` 的太阳直射与高光；
- Fresnel、天空/大气反射、深度吸收和近景波浪法线；
- 消费 M3 的透射与空气透视接口，并从同一静态 `WeatherState` 读取风场输入。

验收：海岸稳定，太空视角无高频闪烁，水面与地形使用一致的太阳和大气光照。

### M5：时间、天体、昼夜与星空

- `TimeState` 区分单调运行时间与可暂停、缩放、跳转的世界时间，并支持确定性重放；
- 简化行星自转模型驱动太阳方向与星空旋转，不实现轨道力学；
- 星表或程序星空、行星遮挡、大气消光和曝光响应；
- 太阳、地形、海洋和大气统一消费动态 `LightingState`。

验收：时间连续推进且可复现；太阳与星空运动一致，星星在白天自然被散射和曝光压制。

### M6：天气数据层与球面云层

- 以时间驱动的天气生产者替换 M3 的静态 `WeatherState` 生产者；
- 球面天气图、风场和有限云壳；
- 体积积分、太阳自阴影、行星阴影；
- 低分辨率渲染和时空重建；大气与海洋直接消费 `WeatherState` 中的气溶胶与风修正。

验收：天气随时间连续且固定 seed 可重放；地表、穿云和太空俯视无平板云层或状态跳变。

### M7：真实地形数据

- DEM 导入、坐标基准转换和离线切片工具；
- 多分辨率缓存、缺失值与海洋掩码；
- 数据来源和许可证元数据。

验收：同一渲染管线可切换程序高度与 DEM，高度源之外无需修改地形渲染器。

### M8：地表覆盖物扩展性验证

- 用少量测试标记或简单实例验证分块加载/卸载事件；
- 验证稳定 seed、表面语义、相机相对实例坐标和独立 LOD 生命周期；
- 不实现正式植被生态、资产和覆盖物渲染功能。

验收：新增覆盖物消费者不需要修改高度源、地形 LOD 或大气核心接口。

## 7. 性能预算与测量

目标基准为桌面独立 GPU、1920×1080、交互相机。每个里程碑记录 GPU、后端、分辨率、可见分块数、采样数、首次 JIT 和稳定帧时间。优化优先级为可见性与 LOD、数据驻留、减少全屏高成本积分，最后才是微小算术改写。

地形、云和大气必须有独立 GPU 时间或可关闭的诊断路径。不得只报告总 FPS 来证明某个子系统性能。

## 8. 配置与可重复性

配置按 `planet`、`camera`、`time`、`celestial`、`terrain`、`surface`、`ocean`、`atmosphere`、`weather`、`clouds`、`lighting`、`space`、`postprocess` 和 `quality` 分组。长度字段包含单位后缀；角度使用 `_degrees`；所有程序生成、天气演化和抖动具有显式 seed。

预览面板修改运行时副本，不隐式覆盖磁盘配置。保存预设必须由明确操作触发，并记录配置版本。

## 9. 暂缓决策

- 地形裂缝最终采用裙边还是拓扑拼接；
- DEM 首个正式支持的数据格式和坐标基准；
- 云的密度表示与时空重建方法；
- 天体系统是否加入轴倾角与简化季节变化；
- 天气系统采用纯程序演化还是允许外部数据驱动；
- 是否在 Taichi kernel 之外使用专门的数据预处理依赖。

这些决策必须由对应里程碑的可运行原型、画面和测量结果驱动。

## M2.5 Terrain/Renderer boundary

The terrain system is the owner of world data and patch residency. For each
frame it returns a data-only `TerrainFrame` containing the desired set, the
fallback-resolved render set, and bounded upload/release operations. It never
imports or calls a renderer.

`TerrainPatchRenderDescriptor` contains only patch identity, global anchor,
screen-space error and edge masks. GPU slots are kept in a separate
`render_slots` mapping because they are renderer resources, not world state.
`PlanetRenderer.apply_terrain_frame()` is the explicit runtime boundary that
applies these operations to GPU slots and updates the render set.

Surface semantics (`SurfaceDescriptor`, stable `SurfaceCellId` and material
weights) live in `surface.py`; terrain geometry and future coverage consumers
can share this contract without coupling to rasterization. `CubeSphereTerrain`
now exposes only the data-only `update(camera, width, height)` contract.

`TerrainRenderer` now owns the Taichi terrain-generation kernels and the
geometry-side slot fields (`offset`, `normal`, `height_m`, material weights and
surface cells). `PlanetRenderer` accesses those fields only through its
explicit `terrain_renderer` component; its own kernels handle camera-relative
transforms, clipping, rasterization, G-buffer writes and compositing.

## M3.1-M3.2 Atmosphere foundation and sky pipeline

The stable atmosphere is a finite spherical shell concentric with the active
`PlanetModel`. `AtmosphereConfig` stores the shell height, density profiles,
inverse-metre optical coefficients and LUT quality, but deliberately does not
duplicate the planet radius. The CPU `AtmosphereModel` is a float64 validation
oracle for shell intersections, Rayleigh/Mie/absorption density, extinction,
transmittance and direct single scattering. It is not a runtime fallback.

The realtime `AtmosphereRenderer` owns a linear-HDR RGB Transmittance LUT. Its
vertical coordinate is normalized distance from the planet-radius cylinder and
its horizontal coordinate reconstructs the distance to the atmosphere top.
Rays hidden by the opaque planet are rejected analytically before lookup. This
distance parameterization preserves substantially more resolution around the
spherical horizon than a uniform zenith-cosine texture.

The linear-HDR Sky-View LUT is parameterized by relative solar azimuth and by
two horizon-centred view-zenith domains. Both the sky and ground halves converge
on the analytical horizon, preventing a single uniform-angle texel from
covering the complete limb gradient. The transmittance texture is rebuilt only
when the planet/atmosphere definition changes. Sky-View is keyed by exact f32
camera radius and local solar zenith plus its lighting inputs; camera yaw alone
does not rebuild it.

GPU geometry stays camera-relative. In the local East-Up-North frame the camera
is the origin and the planet centre is represented analytically as
`(0, -camera_radius, 0)`, where the radius was computed in CPU float64. The GPU
does not receive absolute global positions for atmosphere integration.

Frame composition is now:

```text
G-buffer -> surface lighting -> surface_hdr
         -> sky-view lookup + attenuated finite sun disk -> hdr
         -> exposure -> half-resolution bloom -> tone-map/sRGB -> display
```

`LightingState` stores solar irradiance and angular radius as the only solar
energy inputs. The uniform visible-disk radiance is derived from
`irradiance / (pi*sin(angular_radius)^2)`; it is not an independently
art-directed value. This keeps the disk, atmospheric scattering and every
surface consumer on one radiometric scale. Bloom operates only on the final
composite and never modifies `hdr`; raw transmittance and mask diagnostics
bypass exposure, bloom and tone mapping.

M3.3 adds a view-dependent Aerial-Perspective froxel volume, but the volume is
only a low-frequency cache for RGB in-scattering. It is not authoritative for
camera-to-surface transmittance: opaque pixels reconstruct that value from the
actual G-buffer endpoint with two samples of the spherical Transmittance LUT.
This keeps transmittance at full screen/depth resolution and avoids assigning
different physical meanings to the same trilinearly interpolated froxel depth.

Near the projected planetary limb, optical depth and in-scattering vary too
quickly for the coarse angular froxel grid. A smooth radial-cosine band switches
from cached scattering to a bounded per-pixel integration along the real
camera-to-G-buffer path. Away from that band the cheaper volume remains active.
The hybrid keeps the common case inexpensive while making the horizon path
independent of analytical ground-hit classification and terrain displacement.
The direct path uses two-scale quadrature: base intervals remain concentrated
around the ray's minimum altitude, while intervals near intersections with the
central solar-shadow cylinder receive local substeps. A smooth refinement
weight blends the base and refined estimates, so moving the terminator across
an interval does not expose a second sampling boundary. Finite-disk visibility
still supplies the physical penumbra; the shadow cylinder is used only to find
where that source term needs more samples. Surface rays crossing this narrow
terminator band select the same direct path even when they lie just inside the
ordinary limb band; this prevents the coarse froxel cache from reintroducing
the high-frequency source that the direct quadrature was designed to resolve.
M3.4 adds a static altitude/solar-zenith
Multi-Scattering LUT. Its directional integration estimates the returning
scattered-light factor and sums higher orders as a bounded geometric series;
ground bounce uses the configured Lambertian atmosphere ground albedo.

Before surface shading, the atmosphere computes per-visible-pixel sunlight
transmittance and mean sky radiance from the same LUTs. Formal terrain lighting
therefore uses attenuated `solar_irradiance / pi` plus atmospheric sky light;
the former fixed ambient term is gone. Debug material/LOD/Patch modes remain
unlit so diagnostic colors are not hidden by atmospheric conditions.

Transmittance and Multi-Scattering LUTs rebuild only when their static inputs
change, Sky-View rebuilds after any camera-radius or local-solar change visible
to the f32 GPU kernels, and the scattering-only Aerial-Perspective volume
rebuilds every view frame. Path integration uses variable-width intervals
concentrated around the minimum-altitude point. Direct sunlight uses finite-disk visibility at the
spherical horizon instead of a binary centre-ray shadow. The quality controls
`aerial_horizon_raymarch_steps` and `aerial_terminator_substeps` respectively
set the inexpensive base quadrature and the local refinement, rather than
requiring the expensive count everywhere. Dynamic
weather/aerosol corrections remain later M3 work and must invalidate these
resources through the same ownership path rather than add a second sky
implementation.
