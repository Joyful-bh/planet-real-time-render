# Taichi 实时行星渲染器

这是一个使用 Python 与 Taichi 构建的单行星实时渲染项目。目标是在同一个连续世界中，从贴近地表自由飞行到大气层外，始终正确表现地形、海洋、大气、太阳、云层与星空。

项目不是通用游戏引擎，也不是轨道模拟器。它聚焦于行星尺度坐标、可扩展球面地形、跨高度大气渲染和实时视觉表现。

## 最终体验

- 从地表起飞，无加载画面地穿过云层和大气边缘进入太空。
- 近地表看到有尺度感的山脉、盆地、海岸与水体，不使用无限平面伪装行星。
- 在任意高度看到与观察位置、太阳方向和光学路径一致的大气颜色。
- 白天看到太阳、天空和云；夜晚看到经过大气消光的星空。
- 在太空看到球形地平线、被照亮的行星表面、大气边缘和昼夜分界。
- 地形既可由固定种子的程序噪声生成，也可由真实 DEM 数据提供。
- 支持暂停、加速和指定时间；太阳、星空、云和天气状态随时间连续变化，形成可复现的动态场景。

## 核心技术方向

### 行星与坐标

- 单个球形类地行星。当前路线不支持椭球或扁率模型。
- CPU 侧使用双精度行星坐标；GPU 渲染使用相机相对坐标。
- 采用浮动原点，避免从米级地表移动到万千米尺度时丢失精度。
- 相机使用位置、局部切线基和姿态，不用单一“高度”代替真实位置。

### 地形

- 以立方体球或等价的低畸变球面参数化建立六面分块。
- 使用四叉树 LOD、视锥裁剪、地平线裁剪和屏幕空间误差选择细节层级。
- 高度源使用统一接口，第一阶段实现确定性程序地形，之后接入真实 DEM。
- 分块间必须处理裂缝、法线连续性和 LOD 过渡。

### 海洋

- 海平面是具有真实球率的行星球面，不是局部无限水面。
- 支持 Fresnel、太阳高光、天空/大气反射、深度相关吸收和海岸过渡。
- 近景波浪与远景球面轮廓分层处理，避免为太空视角保留无意义的高频细节。

### 大气

- 球形地球与球壳大气是唯一稳定模型。
- 密度随径向高度变化，包含 Rayleigh、Mie，后续加入臭氧吸收和多次散射近似。
- 地表、云层、大气层顶和太空视角共享同一套物理定义。
- 稳定版本使用 Transmittance、Multi-scattering、Sky-view 与散射型 Aerial-perspective LUT；行星边缘采用有界直接积分，地表透射率根据真实 G-buffer 端点逐像素重建。
- 交互预览提供逐阶段大气诊断视图及相机、地形 LOD、大气 LUT 独立冻结；使用方法见 `docs/ATMOSPHERE_DIAGNOSTICS.md`。
- 太阳是方向光，同时以有限角半径圆盘可见，并参与地形、海洋、大气和云的统一照明。

### 云与星空

- 云使用球面天气场和有限高度云壳，不能绑定局部平面后无限延伸。
- 体积云采用有界 Ray Marching、空区域跳过、提前终止与低分辨率重建。
- 星空使用方向数据或可验证来源的星表；大气内观察时应用消光，白天由曝光和散射自然压制。

### 时间、天气与扩展性

- 统一时间系统只推进世界时间，不直接修改渲染器；天体、天气、云和海洋分别消费时间快照。
- 早期静态太阳和静态天气也使用正式数据接口，后期可替换为动态生产者而不改写渲染消费者。
- 天气系统后期负责球面天气场、云量、风和气溶胶等时变状态，但不演变为通用气象科学模拟器。
- 植被等地表覆盖物当前不实现，但地形分块、表面语义和资源生命周期必须允许后续独立扩展。
- 海洋后期接受太阳、大气、天空和天气风场的统一输入，建立完整水面光照，而不是孤立的颜色效果。

## 辐亮度合成关系

```text
Camera Ray + floating origin
        ├─ Space/stars: background radiance at infinity
        ├─ Terrain/ocean: nearest opaque surface hit
        └─ Clouds: bounded participating-medium intervals
                         ↓
Atmosphere integrates transmittance and in-scattering
along the actual visible ray segments, then composites
background, clouds and nearest surface in depth order
                         ↓
Linear HDR → exposure → bloom → tone mapping → display
```

具体架构和阶段验收见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
关键 Bug、根因和防回归记录见 [docs/IMPLEMENTATION_ISSUES.md](docs/IMPLEMENTATION_ISSUES.md)。

## 当前状态

M0、M1 与 M2 已完成：稳定入口已有 CPU 双精度行星相机、浮动原点、静态 `LightingState` 和 Taichi 自定义光栅管线。地形采用每 patch SSE 驱动的 Mixed LOD（split/merge 滞回、最高 L16、相邻层级差不超过一），并明确分离 Desired、Resident 和 Render Set。Patch 通过优先级队列、固定帧预算、父级 fallback、LRU GPU slot 渐进驻留；转动视角只改变 Render Set。程序高度、顶点和法线由 Taichi GPU kernel 生成，CPU 只传递轻量 descriptor、双精度 anchor 减 camera 后的相对坐标。当前粗细边使用只在必要边启用的 skirt，Geomorph 与真实 DEM 数据加载尚未实现。

## 运行 M0

```bash
python -m pip install -e .
python main.py --backend auto --preview
python main.py --backend cuda --altitude-m 2000000 --pitch-degrees -35 --output output/space.png
```

可以用独立配置直接切换测试尺度，不需要编辑参数文件：

```bash
# 50 km 半径的小行星尺度试验场（带风格化大气）
python main.py --backend cuda --preview --config configs/asteroid.json

# 6360 km 半径、100 km 大气层的类地配置
python main.py --backend cuda --preview --config configs/earth.json
```

`configs/planet.json` 仍是默认配置，当前与 `asteroid.json` 使用相同的
50 km 试验尺度。`asteroid.json` 的大气是用于跨尺度渲染调试的风格化设定，
不表示真实的 50 km 小行星能够维持这样的大气。

预览使用 WASD 沿局部切平面移动、空格径向上升、Shift 径向下降、按住鼠标左键拖动视角。参数面板可以跳转至地表、50 km 和 2000 km 高度，并显示径向高度、解析地平线距离和浮动原点 revision。

## 参考原型（非稳定入口）

仓库当前可运行代码是旧球壳单次散射参考原型，只用于核对球壳求交、Rayleigh/Mie 光学深度和 Taichi 性能。它的局部平面、相机和全屏积分管线不是新架构的稳定入口，禁止在其上继续添加地形、海洋或云功能。

旧大气原型必须显式运行：

```bash
python -m experiments.atmosphere_reference.cli --preset day --backend auto --preview
```

## 非目标

- 多行星系统与行星间切换。
- 轨道力学、航天器动力学或天体仿真。
- 通用游戏场景、实体组件系统、角色和建筑；植被等覆盖物不在当前路线实现，但保留专用扩展接口。
- 完整路径追踪器或离线电影级渲染器。
- 板块运动、侵蚀和天气等完整科学模拟。

## 许可证与数据

许可证将在首次公开发布前确定。真实 DEM、星表、纹理及其他第三方数据必须记录来源、坐标基准、分辨率、处理流程和再分发许可。程序生成结果必须记录算法版本、参数与随机种子。
