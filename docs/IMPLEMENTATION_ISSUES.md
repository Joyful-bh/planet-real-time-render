# 关键实现问题记录

本文件持续记录会影响正确性、跨尺度连续性、性能架构或后续模块接口的关键问题。普通拼写、局部重构和无行为影响的修改不记录。

每个问题至少包含：编号、状态、发现条件、现象、根因、修复、验证和防回归措施。状态使用 `open`、`mitigated` 或 `fixed`；若修复依赖后续里程碑，应明确区分已修缺陷与剩余限制。

## BUG-0001：近裁剪缺失造成大面积表面割裂

- 状态：`fixed`
- 所属阶段：M1 自定义表面光栅化
- 发现条件：交互预览，相机约 934 m 高度，均匀 48×48×6 立方体球，视线接近地平线。
- 现象：画面右侧出现从地平线延伸到底部的大面积黑色缺口；地平线和部分共享边存在漏画风险。

### 根因

旧光栅路径要求三角形三个顶点都位于近裁剪面前方。只要一个顶点位于近裁剪面后，整个三角形就被丢弃。近地表时均匀球面三角形的空间尺寸很大，单个被错误丢弃的三角形可以覆盖大面积屏幕区域。

旧边界测试同时对三条边使用完全闭合的 `lambda >= 0` 判定，没有统一的半开边覆盖约定。共享边浮点结果发生微小偏差时，可能出现相邻三角形都不覆盖同一像素的裂缝。

### 修复

- 顶点变换与投影拆分；先保留观察空间顶点。
- 在投影前按观察空间近裁剪面裁剪三角形。
- 一个顶点在内时生成一个新三角形；两个顶点在内时生成两个新三角形。
- 裁剪交点同步插值观察空间位置与全局单位法线。
- 原始三角形预留两个 raster triangle slot，避免动态分配和不确定写入。
- 光栅边函数统一方向，并使用 top-left 半开覆盖规则。
- tile 分桶改为消费裁剪后的三角形，而不是原始索引三角形。

### 验证

- 地表、约 1 km、高空和太空视角必须不存在贯穿画面的大块背景缺口。
- G-buffer 命中深度必须为正且有限，法线保持单位长度。
- tile overflow 必须为零；否则不能把缺口归因于裁剪。
- 对近裁剪面两侧移动相机，表面覆盖必须连续。

### 剩余限制

均匀 48×48×6 球面在近地表仍会表现出多边形地平线。这不是本 Bug 的残留，而是 M2 分块 LOD 的几何误差问题。top-left 规则防止共享边漏画，但不能替代 MSAA 或后处理抗锯齿。

## BUG-0002：M2 地形错乱、旋转卡死与近地黑色遮挡

- 状态：`fixed`
- 所属阶段：M2 立方体球 LOD 与程序地形
- 发现条件：程序地形启用后从标称参考球面高度 2 m 启动，并在预览中连续旋转视角。
- 现象：地表碎片错乱，近景出现大片纯黑几何，帧率降至个位数；旋转后窗口无响应并可能直接退出。

### 根因

相机高度仍相对无地形参考球面计算，程序高度为正时相机会出生在实体地形内部。跨 LOD 裙边此时处于相机前方，且光栅器未做背面剔除，形成大片暗色遮挡。地形驻留选择又依赖观察方向，纯旋转会在 UI 线程同步生成噪声网格；每次驻留变化还把固定最大容量缓冲全部补零并上传。大量背面和近裁剪后的大三角形进入 tile 列表，使逐像素候选数接近上限，造成严重 GPU 压力。

### 修复

- 初始近地相机及 Surface 按钮改为查询程序地形高度，再增加观察者净空。
- 加入观察空间几何背面剔除，裙边背面不再进入 tile。
- 驻留选择只依赖相机位置；纯旋转不再生成网格或改变驻留集。
- 网格和 patch 锚点改用 ndarray kernel 按活动数量上传，不再传输最大容量补零数组。

### 验证

- CPU 小分辨率实际光栅测试保持有限 HDR，并能命中有效 G-buffer。
- 新增纯 yaw 旋转前后驻留集对象与事件均不变的回归测试。
- 全部 12 项自动化测试通过。实际交互画面、目标 GPU 稳定帧率与长时间旋转由用户验收。

### 防回归措施

相机“参考球高度”和“离真实地形高度”必须明确区分；任何会在预览主线程触发程序网格生成的选择输入都要有稳定性测试。GPU 上传量必须按活动元素统计，背面不得进入 tile 候选表。

### 剩余限制

平移跨越 LOD 驻留边界时仍会同步生成未缓存 patch；后续需要增量后台生成或预算化流送。此处记录的裙边方案已在 BUG-0004 中移除。

## BUG-0003：背面剔除后出现翘起的地形片

- 状态：`fixed`
- 所属阶段：M2 立方体球 LOD
- 发现条件：修复 BUG-0002、启用观察空间背面剔除后，在近地表水平观察。
- 现象：正常地表局部消失，只剩巨大平面片、斜向楔形裂口和看似向上翘起的地形边缘。

### 根因

立方体球参数化的 `du × dv` 指向球外，但主网格长期使用 `(i0, i2, i1)`，其几何法线为反向的 `dv × du`。M1/M2 初版没有背面剔除，因此错误被双面光栅掩盖；BUG-0002 加入剔除后，正常主表面被当成背面丢弃，而绕序不同的裙边和部分裁剪结果仍可见，形成“翘片”。

### 修复

- 主地形三角形统一为外向绕序 `(i0, i1, i2)` 与 `(i1, i3, i2)`。
- 裙边绕序随之统一，测试用 `create_cube_sphere` 采用相同约定。
- 保留背面剔除，不再用双面渲染掩盖拓扑错误。

### 验证与防回归

新增几何测试：重建每个 patch 主表面三角形的双精度位置，要求 `cross(p1-p0, p2-p0) · centroid > 0`。CPU 实际光栅烟雾测试继续要求存在有效 G-buffer 命中。全部 12 项测试通过；最终交互画面由用户验收。

### 剩余限制

本问题修复后的裙边方案仍不正确，已由 BUG-0004 移除并替换为统一层级基线。

## BUG-0004：全边裙边与非共形混合 LOD 造成条带和大面积缺口

- 状态：`fixed`
- 所属阶段：M2 立方体球 LOD
- 发现条件：约 100 km 高度观察同时包含多个层级的自适应 patch 集合。
- 现象：地平线附近出现大片黑色多边形缺口，画面中有跨越数百像素的细长地形条带和墙面。

### 根因

旧选择器在同一帧混合不同四叉树层级，却没有邻接层级平衡、粗细边重采样或 stitch index。为了遮蔽 T-junction，网格生成器又无条件为每个 patch 的四条边创建裙边，包括两个同级常驻 patch 的内部共享边。裙边深度随粗层级可达到数千米，因此会作为真实墙面进入近裁剪、tile 分桶和深度测试；它不能修复不完整的混合 LOD 拓扑，反而把接缝放大成截图中的长条和巨大黑色区域。

### 修复

- 移除全部裙边几何以及对应 GPU 容量开销。
- 根据投影误差先计算目标层级；若可见 patch 超出预算，逐级降低目标层级。
- 每帧只提交一个统一 LOD 层级，保证相邻 patch 边界顶点一一对应。
- 用 patch 中心到四角的最大球面夹角进行保守地平线相交测试，避免递归阶段错误裁掉部分可见 patch。

### 验证与防回归

- 测试要求同一驻留批次内只有一个 `PatchKey.level`。
- 代表高度选择结果：2 m 为 L10/16 块，1 km 为 L8/32 块，100 km 为 L5/44 块，2000 km 为 L3/60 块，均未超过 96 块预算。
- 320×180、100 km、12×12 patch 的 CPU 实际光栅检查：44 patch、12672 三角形、tile overflow 为 0，HDR 有限。
- 全部 12 项自动化测试通过；最终视觉由用户验收。

### 剩余限制

统一层级是正确性优先的稳定基线，不具备局部细节最优性。恢复同帧混合 LOD 前必须实现跨面的邻接查询、最大一级层级差和显式边界拼接；不得再次用“所有边都加裙边”替代拓扑约束。层级切换目前也没有 geomorph/hysteresis。

## BUG-0005：同步整层 LOD 更新导致周期性数秒卡顿

- 状态：`fixed`
- 所属阶段：M2 地形驻留与生成
- 发现条件：相机从太空持续下降并跨越统一 LOD 阈值。
- 现象：每隔一段高度窗口无响应数秒；缓存命中时选择器仍占用约 65–82 ms。

### 根因

旧路径每帧从根遍历并重建统一层级，跨阈值时在窗口线程用 Python 标量 FBM 生成全部 patch，随后 `concatenate` 整批网格并全量上传。实测首次层级构建约 4.6–9.8 秒。

### 修复

- 用持久 Mixed-LOD 叶集合、SSE split/merge 滞回和每次最多固定拓扑变化替代整层切换。
- 拆分 Desired、Resident、Render Set；新增 UNLOADED/REQUESTED/READY/GPU_RESIDENT 状态和父级 fallback。
- 请求按粗 fallback、SSE、距离、视野中心和运动方向排序；生成、上传均有每帧数量预算。
- 删除 TerrainMeshBatch、CPU 网格/法线生成和全量上传；使用固定 GPU slot、共享 topology 和增量 slot 更新。
- 程序高度、cube-sphere 顶点和法线迁移到 Taichi kernel；CPU 只计算 f64 patch anchor 与相机差。
- Render Set 独立执行视锥/地平线裁剪，旋转不改变驻留需求。
- 固定容量缓存使用 LRU 淘汰，并预留 DEM pyramid、异步 I/O、解压与预算上传契约。

### 验证与防回归

测试覆盖 Mixed LOD、高层级、渐进拓扑变化、三集合分离、逐帧上传预算、旋转不改变 Desired、CPU/GPU 高度一致性和实际 G-buffer 命中。首次 GPU kernel JIT 单独统计，不计入稳定帧时间。

### 剩余限制

当前预算按 patch 数量而非精确 GPU 时间；程序 kernel 的首次 JIT 仍较长。DEM provider/cache 只有接口和生命周期骨架，尚无真实数据格式与 I/O。Geomorph 尚未实现，skirt 将在 edge stitching 可用后替换。

## 新问题模板

```text
## BUG-NNNN：标题

- 状态：open | mitigated | fixed
- 所属阶段：
- 发现条件：
- 现象：

### 根因
### 修复或缓解
### 验证
### 防回归措施
### 剩余限制
```

## BUG-0006：Mixed LOD Patch 接缝出现黑色虚线裂缝

- 状态：`fixed`
- 所属阶段：M2 球面分块、程序地形与自定义光栅
- 发现条件：约 2 km 高度，在 Height 或 Patch ID Debug View 中观察同级及粗细 Patch 边界。
- 现象：Patch 交界处出现规则的黑色点状背景像素，移动相机时可能沿接缝闪烁。

### 根因

相邻 Patch 分别使用各自的双精度 Anchor 差和 float32 局部 Offset 重建顶点。理论上相同的球面边界点经过不同的浮点运算路径后并不逐位相等，投影后形成亚像素宽的真实间隙。旧光栅判断 `edge >= -1e-5` 也不是标准 Top-Left 半开覆盖规则，无法可靠处理共享边；同级边界又不会启用只用于缺失邻居的 Skirt。

### 修复

- 根据确定性的 cube-sphere 全局边界坐标建立共享顶点组。
- 顶点变换后执行 GPU edge weld，同级和跨 cube face 的公共顶点复制同一份观察空间位置。
- 对相差一级的边界，在焊接公共端点后把细边奇数顶点投影到粗边折线上，并同步法线、高度和材质插值。
- 屏幕坐标量化到 1/256 像素，光栅覆盖改为 Top-Left 半开规则，删除与三角形尺度无关的魔法容差。
- Skirt 继续只作为缺失邻居或流式 fallback 的兜底，不用于掩盖所有内部边界。

### 验证

- 自动测试检查 weld 操作后的共享观察空间顶点逐位相等。
- 自动测试检查粗细 LOD 拼接顶点位于已焊接端点的中点。
- CPU 小分辨率实际光栅烟雾测试要求 G-buffer 有命中且 HDR 有限。
- 交互 Preview 中的最终接缝视觉检查由用户执行。

### 防回归措施

边界连续性测试必须覆盖同面、跨面与相差一级 LOD。不得用全边 Skirt 或扩大光栅 epsilon 代替几何焊接；修改 Anchor、投影或光栅覆盖规则时必须重新检查 Patch ID View。

### 剩余限制

当前焊接操作表在 CPU 根据 Render Set 构建并上传，规模受固定 Patch 容量限制。后续若 Patch 数显著增加，应缓存未变化的邻接操作表或在 Tile Manager 的驻留事件中增量维护；Geomorph 仍未实现。

## BUG-0007：M2 移动场景帧率低于实时目标

- 状态：`open`
- 所属阶段：M2 Mixed LOD、程序地形与自定义光栅
- 发现条件：交互 Preview 持续移动，约 74 个可见 Patch，桌面窗口高分辨率输出。
- 现象：移动过程中帧率低于 10 FPS，尚未确定 CPU LOD/邻接、GPU Patch 生成或逐像素光栅各自占比。

### 根因

RTX 3050 Laptop GPU、1280×720 的用户基线显示：平均帧时间 185.7 ms，其中 `terrain_dispatch_ms` 为 176.0 ms，而 Taichi 全部 GPU kernel 合计仅约 1.8 ms/帧。Python profile 进一步定位到 `_balance_render_coverage -> _neighbor -> _leaf_at` 的重复扫描、每帧对全部祖先重算 Priority，以及 Coverage 不变时仍重复计算边界 Mask 和可见集。原 `selection_ms` 还会在未选择的帧沿用旧值，造成统计误读。GPU 软件光栅不是当前低于 10 FPS 的主因。

### 修复或缓解

已增加 `tools/profile_m2.py`，分别测量 stable/move 的 CPU dispatch、同步完成的 Patch GPU 工作、Render GPU 工作、P50/P95/P99、Patch/三角形数量、Tile 候选分布、Python call site 和 Taichi kernel 明细。已实施：跳过已请求/驻留 Patch 的 Priority 重算、每帧复用相机裁剪上下文、Desired 祖先缓存、批量 Coverage 平衡、整数 Cube Grid 边界键、Coverage 边界 Mask 缓存、静止相机可见集缓存，以及 Render Set 未变化时跳过 GPU 状态重传。短基准在仍处于 LOD 收敛的情况下由 185.7 ms 降至约 39.2 ms；仍需用户使用完整 stable/move 命令复测。

完成缓存与 BUG-0008 五平面裁剪后，本机 RTX 3050 Laptop GPU、1280×720、78 Render Patch、29,952 个三角形的完全收敛 stable 诊断为平均 15.07 ms（约 66.4 FPS），P95 16.40 ms，Tile overflow 为零；其中 Terrain CPU dispatch 4.57 ms，完整五平面裁剪 GPU kernel 约 0.29 ms。移动场景仍需用户复测，因此本问题保持 open。

### 验证

使用相同 CUDA 设备、配置、分辨率、seed、相机路径和预热帧数保存前后 JSON。首次 JIT 单独排除；`tile_overflow` 必须为零。最终以 Preview 稳定 FPS 和移动 P95/P99 卡顿共同验收。

### 防回归措施

每个重要 M2 性能改动都运行 stable/move 基准，并记录分辨率、后端、LOD 范围、Render Patch 数与 kernel 排名。同步诊断数据不直接等同于正常异步 Preview FPS。

### 剩余限制

当前只有极小 CPU 烟雾结果，不能据此推断目标 CUDA 后端瓶颈或宣称性能改善；需要在用户实际 GPU 上采集基线。

## BUG-0008：高空粗 LOD 三角形产生内部锯齿状黑色缺口

- 状态：`fixed`
- 所属阶段：M2 自定义光栅与 Mixed LOD
- 发现条件：约 85 km 高度、LOD 1–3，视线掠过由大尺度三角形覆盖的地表。
- 现象：Patch 内部出现由长对角线和阶梯边组成的大面积背景缺口；部分最终 Render Patch 边界仍出现细黑缝。

### 根因

旧裁剪器只处理 `z >= 0.1 m` 近面，不处理左右和上下视锥面。跨越相机平面的数十公里级粗 LOD 三角形会在 `z=0.1` 处产生远离视锥的交点，透视除法后屏幕坐标可达数亿像素；float32 边函数发生严重消减并错误判断大片像素覆盖。该形状位于 Patch 内部，因此不是 Horizon/Frustum Patch Culling。参考拓扑全部保持向外绕序，且当前观察变换为反手基底，`front_facing < 0` 符号经验证正确。细缝的附加原因是边界 Mask 根据裁剪前 Coverage 而非最终 Visible Set 计算。

### 修复

- 使用固定容量 Sutherland–Hodgman 多边形裁剪依次处理 near、left、right、bottom、top 五个观察空间平面。
- 最多保留八个裁剪多边形顶点并扇形输出六个三角形；所有位置、法线、材质、高度和 Cell ID 同步插值。
- 仅对完整裁剪后的有限视锥内顶点执行透视除法，避免超大屏幕坐标进入边函数。
- 固定小循环使用运行时上限，避免多层 `ti.static` 导致 JIT IR 爆炸。
- Stitch/Skirt Mask 改为依据最终 Visible Render Set 计算。

### 验证

CPU 实际光栅测试要求所有有效裁剪顶点有限且投影坐标位于视口边界的 1/256 像素容差内，G-buffer 有命中、HDR 有限且 Tile overflow 为零。高空实际画面由用户复测。

### 防回归措施

不得在未裁侧平面的情况下把 near 降至行星尺度三角形不适用的极小值；修改裁剪容量、平面符号、FOV 参数化或投影公式时必须运行投影范围测试。背面剔除符号必须结合坐标基底手性验证，不能通过截图直接翻转。

### 剩余限制

当前没有 far plane；行星地表由 Horizon Culling 控制远端范围。六倍最坏裁剪输出容量增加显存和 `_bin` 扫描上限，后续应配合紧凑 Active Triangle 列表优化，但不得退回不完整裁剪。

## BUG-0009：高速跨尺度移动后 Preview 主线程永久卡死

- 状态：`fixed`
- 所属阶段：M2 LOD Selector 与 Patch Streaming
- 发现条件：启动后跳转至 2000 km，以较大速度下降，并在中低高度持续横向移动。
- 现象：运行一段时间后窗口、FPS 标题和输入同时停止刷新。卡死前常见 GPU Resident 达到 256，历史 READY Patch 大量积累。

### 根因

Selector 的邻接平衡在 Patch 容量边界同时允许“拆粗侧”和“并细侧”，使用无迭代上限的 `while changed`；特定 Mixed LOD 拓扑下可能在两种状态间振荡。Streaming 请求堆的预算只统计有效 Build，不统计失效 heap entry，跨越大量区域后单帧可能无上限清理旧条目。历史 REQUESTED/READY Patch 没有与当前 Needed/Prefetch 集合解绑，所有 READY 又都可参与上传，导致 Slot 满载后继续无效换入换出并放大堆积。

### 修复

- 邻接平衡改为只粗化细侧的单调过程，最多执行 `max_level + 1` 轮；若违反数学收敛条件则明确抛错，不再永久占用 UI 线程。
- `last_changes` 根据最终叶集合是否实际变化计算，避免被撤销的 Split 让 Selector 永久自触发。
- 每帧 heap 扫描同时受有效 Build Budget 和总 Scan Budget 限制。
- heap 超过固定阈值时只保留当前版本的 REQUESTED 记录并重新 heapify。
- 当前 Needed/Prefetch 之外的 REQUESTED 会取消；READY 只有重新进入当前 Needed/Prefetch 才允许上传。
- 历史 READY 继续作为有容量上限的 CPU Cache，不再抢占 GPU Slot。

### 验证

新增无窗口跨尺度压力测试：从 2000 km 连续执行 80 次 24 km 下降，再执行 100 次约 20 km 横移；每帧检查请求堆容量和相邻 LOD 差，整段必须在固定时间预算内完成。完整 CPU/GPU 光栅测试继续验证 G-buffer、裁剪和 Tile overflow。

### 防回归措施

任何 `while` 驱动的 LOD 拓扑修正都必须证明单调量或设置硬上限。异步队列预算必须统计扫描工作而不只统计成功结果；缓存记录、请求堆和 GPU Residency 必须分别设置容量和准入条件。

### 剩余限制

单调粗化可能在 Patch 容量不足时牺牲细节范围，但保持无裂缝和有限完成时间。后续可在 Split 前预估完整一环邻居成本，以减少一次选择中被平衡器撤销的细分。

## BUG-0010: Terrain streaming was directly coupled to renderer resources

- Status: `fixed`
- Phase: M2.5 terrain/renderer boundary refactor
- Symptom: `TerrainTileManager.update()` required a renderer object and
  performed slot uploads, releases and render-set mutation inline. This made
  terrain selection impossible to test or reuse without a raster backend.
- Root cause: patch residency and GPU resource ownership were represented by
  one call path instead of an explicit frame contract.

### Fix

`TerrainTileManager.update(camera, viewport_width, viewport_height)` now
returns a data-only `TerrainFrame`. It contains the desired/resident/render
sets, `PatchUploadRequest` and `PatchReleaseRequest` operations, and a
descriptor-to-slot mapping. `PlanetRenderer.apply_terrain_frame()` is the
runtime adapter that applies those operations. The old renderer-coupled M2
call form was removed after all in-repository callers were migrated.

Surface semantics were moved to `surface.py`, and the height-provider contract
now exposes a small GPU program descriptor. No terrain module imports the
renderer in its implementation path.

### Verification

The M2 regression suite uses the data-only API; CPU raster smoke tests still
produce finite G-buffer/HDR values. The
Taichi terrain-generation kernels now live in a dedicated `TerrainRenderer`
component. The remaining raster backend consumes its geometry fields and owns
camera-relative transformation, clipping, rasterization, G-buffer writes, and
compositing.

## BUG-0011: Tangent camera motion converged toward a pole

- Status: `fixed`
- Phase: M0 planet camera movement
- Symptom: holding `W` or `S` with a non-cardinal heading caused the camera
  path to spiral toward one point on the planet instead of following the
  intended circular tangent orbit. Linear tangent offsets also accumulated a
  small radial error.
- Root cause: preview input was interpreted as a constant local compass
  heading. Recomputing the geographic tangent basis after every step produces
  a rhumb-line path on a sphere; it is not the great circle defined by the
  current camera tangent.

### Fix

`PlanetCamera.move_local()` now separates tangent and radial motion. The
tangent component rotates the camera position around the great-circle normal
using Rodrigues' formula, preserving altitude. The camera forward vector is
rotated by the same transform and converted back to yaw/pitch, which
parallel-transports the heading instead of resetting it to a constant compass
bearing.

### Verification

The camera geometry tests include a diagonal tangent-motion case and verify
constant radius plus a stable great-circle plane. The focused M0 test run
passes (`6 passed`).

### Remaining limitation

Yaw/pitch coordinates remain singular exactly at geographic poles because the
local East/North chart requires a fallback basis. The global position and
great-circle motion remain continuous through that region.

## BUG-0012: Raster kernels scaled with fixed and sparse capacities

- Status: `fixed`
- Phase: M2 raster dispatch/performance
- Symptom: `_transform`, `_clip` and `_bin` processed the configured maximum
  number of slots or clipped-triangle records even when only a small Render
  Set was visible. Back-face rejection also happened after clipping, and the
  pixel rasterizer re-tested every tile candidate for every pixel.
- Root cause: GPU buffers used source-triangle offsets as output positions.
  This left holes between clipped triangles and provided no active-count or
  compact-index contract to later kernels.

### Fix

The renderer now uploads a dense active-slot list and dispatches transform and
clip work from its runtime count. Back-face rejection happens before the
five-plane clipper. Clipped triangles are appended to a dense list using an
atomic counter, so `_bin` iterates only the emitted range. Rasterization uses a
pixel-driven fast path for low tile overdraw; after periodic GPU overdraw
sampling it switches to a triangle-fragment atomic depth pass and one
per-pixel G-buffer resolve when tile candidate counts become high.
The high-overdraw path now consumes a dense tile/triangle-pair stream emitted
by `_bin`; it no longer launches over the full `tiles × 512` candidate
capacity.

### Verification

The M2 CPU G-buffer smoke test passes, and CUDA smoke profiling at 1280×720
shows the compact `_bin` and active transform/clip kernels with zero tile
overflow. The fixed-capacity and sparse-output behavior is documented in
`docs/M2_PERFORMANCE_PROFILING.md`.

The synchronized 180-frame CUDA comparison on the same RTX 3050 Laptop GPU
also exposed an overly eager adaptive-raster switch: the 64-candidate
threshold selected the atomic depth path at only 101 candidates per tile,
raising stable-frame time from 11.98 ms to 19.55 ms and move-frame time from
16.19 ms to 27.20 ms. Raising the switch threshold to 256 keeps the cheaper
pixel path for the current workload; the same run measured 11.98 ms stable
and 16.19 ms while moving, with zero tile overflow.

The follow-up batch-visibility run kept the same workload and reduced the
terrain CPU dispatch to 1.83 ms (stable) and 4.12 ms (moving). Its end-to-end
means were 11.30 ms and 14.48 ms respectively, still with zero tile overflow.
The final dense tile-pair run measured 11.35 ms stable and 14.39 ms moving;
the small difference is normal run-to-run variance.

### Remaining limitation

The tile lookup array still has a fixed overflow cap (`512` candidates per
tile), so extreme overdraw is reported as overflow and requires a future
prefix-sum or hierarchical binning pass for full scalability. Normal frames
no longer scan that unused capacity on either raster path.

The final moving profile still has a long tail (P99 frame 63.72 ms; P99
`terrain_upload_dispatch_ms` 45.86 ms) despite a 14.39 ms mean. The GPU
raster kernels take roughly 2.5 ms per frame in total in this workload, so these stalls
are an upload/driver or CPU streaming scheduling problem rather than a
triangle coverage problem; a future timeline/fence pass must isolate it
before increasing terrain density.
