# M2 性能诊断与优化路线

## 目标

移动时低于 10 FPS 不能只用窗口标题判断根因。M2 需要分别测量：

1. LOD Selector、邻接、可见性和 Render Set 拼装的 CPU 时间；
2. 新 Patch 高度、顶点和法线生成完成所需的 GPU 时间；
3. 稳定 Render Set 下自定义光栅管线的 GPU 时间；
4. 移动场景的平均帧时间以及 P95/P99 卡顿；
5. Patch、三角形和每 Tile 候选数量是否超出合理范围。

首次 JIT 不计入稳定帧结果。诊断脚本在预热后清空 Taichi kernel profiler。

## 使用方法

在项目根目录运行：

```powershell
python tools/profile_m2.py --backend cuda --scenario both --warmup-frames 60 --frames 180
```

固定预热结束后，脚本还会等待 `last_changes == 0` 且 REQUESTED/READY 队列清空，再开始计时；`--settle-max-frames` 控制最长等待。输出中的 `settle frames` 非零是正常的，说明固定预热帧不足以完成渐进 LOD 驻留。

检查分辨率缩放：

```powershell
python tools/profile_m2.py --backend cuda --scenario stable --width 640 --height 360 --frames 120
python tools/profile_m2.py --backend cuda --scenario stable --width 1280 --height 720 --frames 120
```

保存结构化结果：

```powershell
python tools/profile_m2.py --backend cuda --scenario both --json output/m2_profile.json
```

脚本有意在 Terrain Update 和 Render 后分别执行 `ti.sync()`。这会扰动流水并降低吞吐量，因此数据用于归因，不代替 Preview 的最终 FPS。重点比较同一机器、后端、配置下各阶段和不同版本的相对变化。

## 指标解释

- `terrain_dispatch_ms`：Python LOD、集合、邻接、可见性、操作表构建和 kernel 提交。
- `terrain_gpu_ms`：等待当帧 Patch 生成/上传完成的时间。
- `render_dispatch_ms`：提交相机变换、焊接、裁剪、分桶、光栅和显示 kernel 的 CPU 开销。
- `render_gpu_ms`：等待完整渲染完成的时间。
- `selection_ms`：Selector 自身耗时，不包括 Tile Manager 的集合处理。
- `tile_overflow`：必须为零；非零表示某些 Tile 丢失三角形，性能数据也不再代表正确画面。
- P95/P99 明显高于 P50：通常是 LOD 更新、Patch 生成或邻接重建造成的移动卡顿。

## 当前最可能的瓶颈

### 1. 自定义软件光栅的逐像素候选循环

当前 `_raster` 对全分辨率每个像素遍历所在 Tile 的全部候选三角形。在 2560×1365 等高分辨率下，像素数约为 720p 的 3.8 倍；如果地平线附近大量三角形落入相同 Tile，开销还会进一步放大。若 `render_gpu_ms` 主导且随像素数近似线性或更快增长，这就是首要瓶颈。

优先改进：

- 统计 Tile 候选数量的平均值、P95 和最大值，而不只统计 overflow；
- 将 `_clear`、背景着色和表面输出减少为更少的全屏 pass；
- 只遍历紧凑 Active Patch/Active Triangle 列表，禁止每帧扫描全部 GPU Slot 容量；
- 增加内部渲染分辨率比例，先以 0.5–0.75 倍渲染，再使用深度感知上采样；
- 后续评估深度预通过、层级 Tile 剔除或迁移到硬件光栅 API。Taichi 自定义光栅应保留为明确可测的实现选择，而不是无上限堆叠逐像素工作。

### 2. 固定容量扫描而非活动集合扫描

`_transform`、`_clip` 及其缓冲按 `max_gpu_patches` 上限分配，部分 kernel 仍会扫描完整容量。截图中实际 Render Patch 约为 74，而容量为 256；这可能浪费约 3.5 倍的顶点和三角形调度。

优先改进：建立紧凑 `active_slots` 和 `active_triangle_count`，所有几何 kernel 只遍历实际 Render Set。固定 Slot 仍用于驻留，渲染调度不应等同于驻留容量。

### 3. 移动时的 Patch 生成与邻接重建

每个新 Patch 会运行程序高度和法线 kernel。高度函数包含多组 3D FBM；即使每帧只上传四块，也可能形成明显 GPU 峰值。Render Set 变化还会重新构建边界 weld/stitch 表。

优先改进：

- 根据实测 GPU 毫秒而不只是 Patch 数量控制生成预算；
- 将 Patch 生成放入独立队列，允许跨帧完成并在 fence 完成后切换；
- 缓存 Render Set 邻接关系，只对新增、删除和 Slot 迁移增量更新；
- 测量预计算 3D 噪声纹理相对内联 FBM 的速度、显存和画质后再决定是否进入稳定路径；
- 提高运动方向预取有效率，减少相机到达后才生成 Patch 的情况。

### 4. CPU Selector 和 Python 集合操作

若 `terrain_dispatch_ms` 或 `selection_ms` 主导，应优化 `_neighbor`、`_balance_render_coverage`、可见性和 descriptor 构建。优先使用持久邻接索引、按变化增量维护 Desired/Resident/Render Set，避免每次选择对所有 Patch 重复计算方向、边界球和相机基。

## 实施顺序

1. 用脚本取得 CUDA 下 stable/move 基线，并记录分辨率、Patch 数和 P50/P95/P99。
2. 增加紧凑 Active Slot 调度，这是不改变画质且风险较低的结构优化。
3. 增加 Tile 候选分布统计，再针对 `_bin/_raster` 优化。
4. 根据移动场景数据决定优化 GPU 程序高度还是 CPU 邻接/选择器。
5. 最后引入动态内部渲染分辨率；它是质量与性能策略，不能掩盖 overflow 或同步卡顿。

每项优化都必须使用相同配置、相机路径、seed、分辨率与后端进行前后对比，并将首次 JIT 与稳定时间分开报告。

## M2.5 boundary timings

After the terrain/renderer split, `terrain_dispatch_ms` measures only the
data-only terrain update. `terrain_upload_dispatch_ms` measures applying the
returned slot operations to the renderer. `terrain_gpu_ms` still measures the
synchronized completion of patch generation and upload kernels. Keeping these
values separate prevents GPU resource work from being misattributed to the
LOD selector.
