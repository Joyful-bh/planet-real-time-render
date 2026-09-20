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
The high-overdraw path directly rasterizes compact clipped-triangle screen
bounds and no longer consumes the bounded per-tile candidate list. The
low-overdraw path keeps its fast 512-entry tile list, but any overflowing tile
is recomputed from the complete triangle stream in the same frame before
shading. Overflow therefore selects a slower correctness fallback instead of
discarding geometry.

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

### High-density correctness follow-up

Reducing terrain SSE to `4/2` raised the visible set to 172 patches and about
231,000 triangles. Horizon tiles exceeded 512 candidates, and both former
raster paths consumed the same truncated list, leaving irregular black holes.
The direct atomic path now has no per-tile capacity. The pixel path launches a
GPU-only overflow repair for affected tiles without a CPU synchronization.
The true maximum candidate count and overflow total are retained as diagnostics
and shown in Preview; they no longer imply missing triangles.

The focused CPU smoke test compiles both raster strategies and produces a
finite G-buffer. High-density CUDA visual verification remains user-run.

An earlier moving profile had a long upload tail (P99 45.86 ms). It was
isolated to repeatedly marshalling several external arrays into the terrain
generation kernels, rather than GPU terrain computation. The persistent
device staging descriptor described below resolves that stall.

### 2026-09 performance follow-up

Profiling a later renderer revision confirmed that normal tiles were again
entering the atomic depth path after its threshold had been reduced to 64.
The threshold is restored to 256, and both raster variants are compiled during
startup so a later adaptive switch cannot introduce a one-second interactive
JIT stall.

CPU patch visibility is vectorized again and immutable patch anchors/bounds
are cached. Renderer preparation now submits anchor upload, buffer clearing
and vertex transformation together; welding and mixed-LOD stitching are also
submitted together. Patch upload descriptors use one persistent device
staging field rather than passing several external arrays to three kernels.

The synchronized RTX 3050 Laptop GPU comparison at 1280x720 measured 10.86 ms
stable and 13.33 ms moving before the final staging change, corresponding to
92.1 and 75.0 FPS. Stable frame times no longer contain the adaptive-raster
JIT spike. A focused 60-frame moving check after device staging reduced upload
P99 from 47.14 ms to 2.52 ms and the maximum from 66.98 ms to 3.06 ms. Visual
continuity remains a user-run preview acceptance check.

## BUG-0013: High-speed low-altitude flight caused streaming-frame stalls

- Status: `fixed`
- Phase: M2 terrain streaming/performance
- Symptom: sustained flight near 4 km altitude at roughly 12 km/s could fall
  to about 14 FPS even though only 40--80 patches were visible.
- Root cause: rapidly changing render sets rebuilt patch-edge identity through
  Python dictionaries and repeatedly marshalled several variable or
  maximum-capacity edge arrays. Once residency filled, upload selection also
  scanned every cached READY record and performed a full resident-set scan for
  each individual eviction. These costs were outside the terrain manager's
  UI `upload_ms`, so the panel misleadingly reported sub-millisecond uploads.

### Fix

Patch edge identities are cached and welded through a vectorized sort. Weld
and stitch data share one compact operation stream containing only live
records. Prefetch ranking is refreshed with the LOD selection rather than on
every frame. READY selection now iterates the small active-request set, and a
batch LRU eviction performs one resident scan for the whole upload budget.
The profiler accepts an explicit initial altitude so this flight regime is a
repeatable benchmark rather than being represented by the old 25 m/frame
surface test.

### Verification

At 1280x720 on the RTX 3050 Laptop GPU, the targeted 3.85 km altitude,
200 m/frame test improved from 26.59 ms (37.6 FPS) to 14.71 ms (68.0 FPS)
before the final batch-eviction change. GPU rendering remained about 7--8 ms;
the improvement came from removing host-side streaming work. The M2 terrain
tests pass. Interactive visual validation remains user-run.

### 2026-09 landform-fidelity performance follow-up

Lowering SSE to 4/2 and enabling `procedural_landforms_v1` raised the moving
2 km workload to 138--183 visible patches and roughly 185k--246k submitted
triangles. A synchronized 60-frame diagnostic initially measured 87.60 ms per
frame (11.4 FPS). GPU residency was only 342--378 of 512 slots, proving that
the regression was not caused by a full GPU Patch cache. The dominant costs
were repeated CPU leaf-neighbor lookup, unused per-descriptor SSE evaluation,
render-edge rebuilding and 16x16 tile candidate scans.

Render descriptors now contain only data consumed by the renderer. Boundary
masks use one compact integer leaf index per set, fixed edge topology is
cached, edge-operation staging uses one allocation, and selector metadata has
a fixed 2048-entry default bound. Raster tiles are 8x8, reducing the last-frame
mean candidate count from 17.05 to 7.03 without overflow. Per-camera SSE values
are cached for the current update.

The same synchronized workload then measured 45.82 ms (21.8 FPS) with terrain
updated every render frame. With the preview's decoupled 1-in-2 terrain cadence
it measured 24.05 ms (41.6 FPS), a 16.23 ms median and no tile overflow. Pure
CPU terrain selection now runs on a single back-pressured worker in Preview;
the main thread applies only the newest completed incremental frame and never
queues stale camera snapshots. Visual smoothness of this asynchronous path
remains an interactive user acceptance check.

## ISSUE-0014: Procedural terrain logic was coupled to rendering

- Status: `fixed`
- Phase: M2 terrain architecture
- Symptom: changing the procedural terrain algorithm required coordinating a
  CPU height provider, a separate GPU descriptor and renderer-local Taichi
  noise functions. These implementations could silently diverge.
- Root cause: the renderer owned terrain-generation equations while the world
  data layer owned a second CPU implementation; the height-source contract
  described data but not executable terrain behaviour.

### Fix

`TerrainHeightModel` now defines the renderer-independent height contract.
`FbmTerrainGenerator` owns its typed config plus adjacent CPU and Taichi
sampling implementations. `TerrainRenderer` receives a model explicitly and calls
its GPU sampler. The old provider, source and GPU descriptor APIs were removed
rather than retained as compatibility aliases. Inputs are normalized global
directions and outputs are radial heights in metres.

### Verification

The source tree compiles, old API identifiers are absent, and the focused M2
CPU-render test passes while comparing generated GPU-path vertex heights with
the CPU model. Full visual acceptance remains user-run.

## ISSUE-0015: Terrain algorithm parameters leaked into application config

- Status: `fixed`
- Phase: M2 terrain architecture
- Symptom: the application configuration and construction sites knew the
  current procedural algorithm's `seed`, while adding or replacing algorithm
  fields required edits outside the algorithm module.
- Root cause: there was no distinction between selecting a terrain algorithm
  and validating that algorithm's private parameter schema.

### Fix

The external `TerrainConfig` is now a thin envelope containing only a stable,
versioned generator ID and an uninterpreted parameter mapping. An explicit
Registry and Factory resolve `procedural_fbm_v1`, construct its private
`FbmTerrainConfig`, and return `FbmTerrainGenerator`. The renderer receives
only the resulting `TerrainHeightModel`. The general height contract also no
longer requires a `seed`; stable surface-cell seeds are derived independently.
The model reports a conservative height range so visibility code no longer
contains the former FBM-specific 10 km relief constant.

### Verification

Configuration loading and factory construction were checked with the project
JSON. Focused tests cover valid conversion, unknown generator IDs and invalid
algorithm parameters. Source compilation passes; visual acceptance remains
user-run.

## ISSUE-0016: Landform semantics and LOD detail were not represented

- Status: `fixed`
- Phase: M2 procedural terrain fidelity
- Symptom: the reference FBM produced height but no stable distinction between
  oceans, plains, mountain belts, plateaus, basins and canyons. Formal rendering
  reused the height-debug palette, normals exposed low-order mesh derivatives,
  and low-altitude terrain could remain visibly polygonal after the selector
  claimed to have converged.
- Root cause: LOD error was derived only from sphere curvature. The 2:1
  balancing pass also coarsened every newly refined patch next to a coarse
  neighbor. A following merge could remove a required transition-ring patch
  and recreate it during balancing in the same update, leaving the final set
  unchanged despite large remaining SSE.

### Fix

`procedural_landforms_v1` now combines a warped continent/ocean field with
regional mountain belts, flat plateau provinces, basins, canyon contours and
frequency-separated local detail. CPU and Taichi samplers return height plus
normalized semantic weights. Each generator reports unresolved local relief,
which is combined with sphere curvature for SSE and used by request priority.
The default target is reduced to 4 px split / 2 px merge.

Balanced splits now recursively build only the minimal coarse-neighbor
transition ring under the existing per-update and leaf budgets. Merge rejects
operations that would violate the 2:1 invariant. Fully hidden patches can
remain coarse behind the spherical horizon while resident parents retain
coverage. Patch normals use fourth-order centered mesh differences where a
full stencil exists and share the existing welded edge result at boundaries.

Formal surface rendering is separate from the height, LOD and Patch ID debug
views. It combines semantic material weights, radial slope, a shared global
sun direction and shared solar irradiance. Ocean, fertile ground, arid ground,
rock and snow therefore remain readable without coupling the terrain generator
to final RGB colors.

### Verification

Focused registry, landform-weight, terrain-frame and mixed-LOD tests pass. The
small CPU G-buffer test passes both raster paths and verifies finite output.
CUDA images were rendered after LOD settling at 100 km (space), 20 km, 2 km
clearance and 120 m clearance into `output/landform_checks_final/`. Inspection
showed continuous coastlines, regional mountain/lowland structure, finer
low-altitude silhouettes and no new black holes, Patch cracks or normal flips.

### Remaining limitation

The 50 km test planet intentionally exaggerates relief, and the fixed 256-leaf
budget can become the active constraint before every patch reaches the 4 px
target. Negative-elevation terrain is currently colored as water but remains
terrain geometry; a level spherical water surface and water lighting belong to
the later ocean milestone. The procedural model does not yet simulate erosion
or drainage networks, so canyon contours are deterministic shape fields rather
than hydrological rivers.

## ISSUE-0017: Sky, sun and display composition were coupled to surface shading

- Status: `fixed`
- Phase: M3 atmosphere architecture
- Symptom: the surface shading kernel directly emitted the space background,
  an unattenuated solar disk, final HDR and display-encoded color. A spherical
  participating medium could not be inserted without mixing atmosphere math
  into the terrain renderer or overwriting the original surface radiance.
- Root cause: M1 established a deliberately compact terminal shading pass, but
  the frame had no explicit linear-HDR boundary between opaque surfaces,
  atmosphere composition and post-processing.

### Fix

The opaque pass now writes `surface_hdr`. An independent
`AtmosphereRenderer` owns Transmittance and Sky-View LUTs, composites physical
Rayleigh/Mie single scattering and the atmosphere-attenuated finite solar disk
for background pixels, then writes final `hdr` and display output. The CPU
float64 `AtmosphereModel` provides the reference shell, density, optical-depth
and single-scattering equations. Runtime GPU geometry uses camera radius and a
local analytical planet centre rather than absolute float32 planet positions.

### Verification

Focused CPU tests cover inside/outside sphere roots, opaque-ground interval
termination, non-negative density/scattering and bounded transmittance. A
small CPU Taichi render compiled both raster strategies plus both atmosphere
LUT kernels and produced finite G-buffer/output data. Interactive validation
at the surface, horizon, 20 km and space remains user-run.

### M3.3-M3.4 completion

Opaque terrain now consumes a low-resolution Aerial-Perspective volume storing
cumulative RGB scattering, while camera-to-surface transmittance is
reconstructed per pixel from the 2-D Transmittance LUT. The sky and aerial integrators
also sample a static Multi-Scattering LUT with bounded higher-order feedback
and Lambertian ground bounce. Terrain direct light is attenuated along the
surface-to-sun path, while its former fixed ambient term is replaced by mean
atmospheric sky radiance from the same multiple-scattering solution.

The remaining approximation is LUT resolution and the production-oriented
isotropic higher-order closure; it is not a second atmosphere model. Dynamic
weather/aerosol modulation and a dedicated high-quality surface irradiance LUT
remain future extensions.

## ISSUE-0018: Full-resolution aerial ray marching would not scale

- Status: `fixed`
- Phase: M3 atmosphere performance architecture
- Symptom: applying in-scattering and extinction separately to every opaque
  pixel would add tens of atmosphere samples at display resolution, consuming
  the performance budget needed by later ocean and cloud stages.
- Root cause: the G-buffer exposed exact surface distance, but there was no
  reusable view-space representation of integrated atmosphere transport.

### Fix

A small camera-frustum Aerial-Perspective LUT integrates each view ray once and
stores cumulative radiance over normalized slices of its actual atmospheric
segment. The common-case full-screen pass uses trilinear reconstruction, while
the planetary-limb band performs a bounded direct integration. Surface
transmittance uses two full-resolution 2-D LUT lookups instead of the froxel.
Static transmittance and multiple scattering are shared by Sky-View, aerial
perspective and surface lighting instead of recomputing long sun paths.

### Verification

The focused Taichi CPU smoke render compiles all four LUT paths, checks finite
non-negative scattering, bounded transmittance, non-zero sky/multiple/aerial
radiance and finite final HDR output. High-resolution CUDA timing and visual
acceptance across the four canonical altitudes remain user-run.

### Remaining limitation

The aerial volume currently rebuilds every frame and uses fixed configured
spatial dimensions. Temporal reprojection and adaptive quality are deferred
until profiling demonstrates that this low-resolution pass is material.

## ISSUE-0019: Atmosphere discontinuities could not be isolated by stage

- Status: `fixed in code; cross-scale visual acceptance pending`
- Phase: M3 atmosphere validation
- Symptom: low-altitude views can show a dark horizontal band, the apparent
  horizon can move in discrete steps during vertical travel, and the daytime
  sky is difficult to evaluate independently from exposure, terrain LOD and
  final surface composition.
- Suspected causes: quantized Sky-View invalidation, insufficient sampling near
  the spherical horizon and planet-shadow boundary, low-order multi-scattering
  approximation, and uncalibrated small-planet optical depth.  A separate
  terrain silhouette change remains possible because terrain Geomorph is not
  implemented.

### Diagnostic baseline

The preview can now display reconstructed Sky-View radiance, camera-ray
transmittance, raw Transmittance and Multi-Scattering textures, Aerial-
Perspective scattering/transmittance and the binary G-buffer surface mask.
View-dependent atmosphere LUT generation, terrain LOD updates and camera input
can be frozen independently.  The panel reports the captured Sky-View altitude,
camera yaw/pitch, freeze state and rebuild counters.  This makes atmosphere
updates distinguishable from opaque geometry changes without introducing a
second rendering path.

### Verification

Sky-View no longer uses altitude or solar-angle buckets: its key contains the
exact camera radius and local solar cosine representable by the f32 kernels.
Transmittance now uses the distance-to-top spherical mapping and rejects
ground-blocked rays analytically. Sky-View concentrates angular samples on
both sides of the analytical horizon. Transmittance, sky, multiple-scattering
and aerial integrations use variable-width intervals concentrated around the
minimum-altitude point, and finite solar-disk visibility softens the
planet-shadow boundary. Generated texel centres and lookup coordinates now use
the same half-texel convention.

Source compilation and the small CPU render smoke test pass. The smoke test
moves the camera by two metres, below the former altitude bucket, and verifies
that frozen LUTs remain unchanged and resume with an immediate Sky-View/Aerial
rebuild. It also checks every diagnostic view for finite bounded display
output. Cross-scale visual acceptance using the procedure in
`docs/ATMOSPHERE_DIAGNOSTICS.md` remains user-run.

### Remaining work

Physical atmosphere calibration, terrain morphing and ocean geometry remain
separate work. Multi-Scattering remains a bounded low-order approximation, and
the finite-disk optical-depth sample uses one representative visible direction
rather than integrating many samples over the solar disk. These limitations
must not be hidden with exposure or post-processing.

## ISSUE-0020: Formal surface shading was discontinuous at Patch boundaries

- Status: `fixed in code; visual acceptance pending`
- Phase: M2/M3 surface integration
- Symptom: stable ledges or broad shading discontinuities appeared along some
  same-level and mixed-LOD Patch boundaries. The Surface Mask, Height and LOD
  diagnostic views remained continuous, and freezing terrain LOD left the
  artifact unchanged.
- Root cause: the edge-fix kernel treated generated resident attributes as
  mutable frame data. Welding and stitching overwrote persistent normals,
  heights, materials and cell IDs in GPU Patch slots. Those edits survived
  later render-set and adjacency changes until a Patch happened to be rebuilt.
  In addition, welding copied one Patch's one-sided boundary normal to its
  neighbor instead of constructing a common normal from all incident Patch
  estimates. Unlit diagnostic modes hid both errors, while formal directional
  lighting made the normal discontinuity prominent.

### Fix

The renderer now copies resident attributes into explicit frame-local normal,
height, material and cell streams during frame preparation. Edge welding and
mixed-LOD stitching modify only those streams and the frame-local view
positions; generated Patch slots remain immutable. Each exact shared-vertex
group accumulates the normals from all incident Patches, normalizes the common
result and writes that identical normal back to every member. Stitched fine
vertices then interpolate the already-welded endpoints. Clipping and
rasterization consume only the resulting frame-local attributes.

### Verification

The focused CPU render regression compiles both raster paths, verifies exact
shared positions and matching shared normals, and asserts that resident normal,
height, material and cell fields are byte-for-byte unchanged after rendering.
The test passes. Visual validation at the originally reported boundary remains
user-run.

## ISSUE-0021: Solar disk and atmosphere used unrelated radiometric scales

- Status: `fixed in code; visual calibration pending`
- Phase: M3 atmosphere radiometry and post-processing
- Symptom: the daytime sky remained dark outside the forward Mie lobe, while
  the solar disk looked like a separate flat white object with a clearly
  detached halo. The documented HDR pipeline included Bloom, but the runtime
  path performed only exposure, an ACES fit and sRGB encoding.
- Root cause: `solar_irradiance` drove scattering and surface lighting while an
  independently configured `sun_disk_radiance` drove the visible disk. Their
  ratio did not equal the finite disk's projected solid angle, so the two representations
  of the same sun could not be energy-consistent. Display conversion also
  lived inside atmosphere composition, leaving no real post-processing stage.

### Fix

`LightingState` now owns only solar irradiance and angular radius. Uniform disk
radiance is derived from `E = L * pi*sin(alpha)^2`, making terrain,
atmosphere and the visible sun consume one energy scale. Configuration version
3 removes `lighting.sun_disk_radiance` and places exposure plus Bloom controls
in a dedicated `postprocess` group.

Atmosphere composition now stops at an untouched linear `hdr` buffer. A
dedicated `PostProcessor` applies exposure, soft-knee bright-pass extraction,
configurable separable half-resolution Gaussian passes, ACES tone mapping and sRGB
encoding. Bloom is enabled only for the final composite, so it joins the
finite disk to its physical Mie aureole without contaminating Sky-View or LUT
diagnostics. Raw transmittance and surface-mask views bypass the complete
display-lighting transform.

### Verification

Unit tests verify that integrating derived disk radiance over its projected solid angle
recovers the configured RGB irradiance. The configuration-v3 file loads, all
changed sources compile, and the focused 64x48 CPU Taichi render passes both
raster paths, atmosphere diagnostics and the new post-processing kernels.
Interactive exposure/Bloom calibration at the surface, 20 km and space remains
user-run.

### Remaining limitation

This establishes a consistent scale but does not yet replace the bounded
isotropic Multi-Scattering approximation or implement automatic camera
exposure. Those should be changed only if fixed-view HDR probes show an energy
deficit after visual validation, rather than compensating with arbitrary sky
offsets.

### Visual acceptance follow-up

The first Bloom implementation failed visual acceptance: the physically
derived solar radiance (about tens of thousands in the current HDR scale) was
fed unbounded into a small finite-support blur. Most of that rectangular
support tone-mapped to white, enlarging the sun into a nearly square bright
block instead of producing a decaying glare tail. The disk itself also used a
binary pixel-centre test, so its geometric edge had no subpixel coverage.

The finite disk now uses analytical angular distance with a one-pixel smooth
coverage filter. Bloom bright-pass values retain their RGB ratio but are
bounded before convolution to the range represented by the realtime kernel;
four low-energy Gaussian passes then create a soft tail without redefining the
disk silhouette. This correction compiles and passes the focused 64x48 CPU
render regression; interactive visual acceptance remains pending and the
earlier failed appearance is not considered a successful P3 result.

## ISSUE-0022: Earth-scale aerial perspective produced dark froxel bands

- Status: `fixed in code; visual acceptance pending`
- Phase: M3 aerial perspective / cross-scale continuity
- Symptom: the Earth-radius preset showed broad black spots and horizontal or
  vertical dark bands on the unlit hemisphere. The artifacts were isolated in
  the Aerial Scattering diagnostic and followed the coarse froxel grid rather
  than terrain Patch or LOD boundaries.
- Root cause: every view ray shared a depth axis derived from the camera's
  global horizon distance, with cubic spacing concentrated near the camera.
  From an Earth-scale space view this range spans millions of metres, although
  scattering occurs only in the roughly 100 km atmosphere segment near the
  ray's far end. Only a few of 32 depth slices therefore represented active
  medium, and trilinear reconstruction enlarged their transitions into bands.
  Angular generation used texel centres while reconstruction used an endpoint
  coordinate convention, adding a half-texel spatial offset.

### Fix

Each aerial froxel ray now parameterizes depth from its own atmosphere entry to
its horizon-continuous prefix endpoint: the first ground intersection for a
ground hit, closest approach for a grazing miss, or atmosphere exit for an
outward ray. All depth layers therefore sample active medium independently of
planet radius and camera distance.
Runtime sampling reconstructs the same normalized per-ray coordinate, and the
angular axes now use the matching half-texel convention. The Earth preset uses
a 96 x 54 x 64 aerial volume; the small-planet preset retains its cheaper
quality setting.

### Verification

Configuration parsing, Python compilation and the focused CPU atmosphere smoke
render cover the new kernel signature and finite output. Visual acceptance of
the previously reported Earth-scale dark-side view remains user-run.

### Remaining limitation

The froxel volume is still view dependent and rebuilt every active frame.
Adaptive angular resolution and temporal reconstruction remain performance
work; they must not reintroduce a planet-scale physical-distance depth axis.

### Horizon-continuity follow-up

The first per-ray implementation still used atmosphere exit as the endpoint
for a ray that narrowly missed the reference sphere. Its immediate neighbor
could narrowly hit the sphere and use the near ground intersection instead.
Although both choices were individually valid, their normalized depth axes
were topologically discontinuous at the horizon. Angular interpolation mixed
unrelated depths, producing intermittent bright/dark spots in Aerial
Transmittance and visible stepwise changes during camera motion.

Grazing misses now end at their forward closest-approach point. As a ground
intersection approaches tangency, its near root converges continuously to that
same point. This keeps neighboring froxel depth coordinates compatible at the
silhouette without integrating through the planet or reverting to a global
camera-distance axis.

### Production-path refactor after incomplete visual acceptance

Visual testing showed that endpoint continuity alone was insufficient. The
remaining scallops repeated at roughly 13 pixels, matching the Earth preset's
`1280 / 96` and `720 / 54` aerial-froxel footprint. Near a grazing ray,
transmittance is exponential in optical depth and changes much faster than the
uniform angular grid. More fundamentally, the same normalized depth still
represented different physical distances and altitudes in adjacent rays, so
ordinary trilinear interpolation was not physically meaningful. The analytic
reference sphere could also classify a ray as a miss while displaced terrain
still produced a G-buffer hit.

The aerial volume is now scattering-only. Camera-to-surface RGB transmittance
is reconstructed per pixel from the actual G-buffer distance using the ratio of
two unoccluded Transmittance-LUT samples along the reverse ray. This removes
angular froxel interpolation from the quantity where the artifact was most
visible. Scattering keeps the inexpensive volume in low-frequency regions, but
smoothly switches to bounded per-pixel integration when the surface radial/view
cosine enters a configurable limb band. The direct path integrates to the real
terrain endpoint, not an analytic ground intersection.

Earth-scale sphere roots now evaluate the discriminant as
`(radius - perpendicular_distance) * (radius + perpendicular_distance)` instead
of subtracting two squared million-metre quantities. The old 3-D aerial
transmittance field and its internal API were removed rather than retained as a
second, inconsistent source of truth.

New quality controls are `aerial_horizon_raymarch_steps`,
`aerial_horizon_inner_cosine` and `aerial_horizon_outer_cosine`. The two cosine
values define the smooth direct-to-froxel transition; they are not angular
resolution multipliers.

The focused CPU render regression compiles both raster paths and every
atmosphere diagnostic with the refactored kernels, checks finite HDR/display
output and bounded transmittance, and passes. CUDA performance and visual
acceptance at the reported Earth-space and low-altitude views remain user-run.

### Terminator-scattering follow-up

After full-resolution camera-to-surface transmittance passed visual acceptance,
the Earth preset still showed smaller bright blocks in `Aerial scattering`,
especially where the distant limb met the dawn/dusk terminator. Raising the
fixed direct-integration count from 16 to 48 only reduced their size. This
isolated a second error: a long grazing path used deterministic midpoint
quadrature, while finite-disk sunlight and sun-path optical depth can change
inside a much narrower interval. Neighboring rays therefore moved whole
midpoint samples between shadow and light.

The direct limb integrator now intersects each camera ray with the central
solar-shadow cylinder and treats those roots as sampling features, not binary
visibility. It retains altitude-warped base intervals, but continuously blends
each base estimate with a locally subdivided estimate near a shadow root or
inside the finite-disk penumbra. This makes the quadrature estimate continuous
as the feature crosses an interval boundary and avoids paying the refined cost
over the complete atmosphere segment. A narrow, smoothly faded surface-path
band around the same roots also selects direct integration outside the ordinary
limb-cosine band, so the low-resolution froxel cache cannot reintroduce the
terminator discontinuity. `aerial_terminator_substeps` controls the local
refinement; the Earth base count is restored to 16 with eight local substeps
instead of retaining the diagnostic 48-step global march.

Source compilation and a focused CPU kernel smoke test cover the new control,
shadow-root feature calculation and finite output. Final CUDA visual acceptance
at an Earth-scale space terminator remains user-run.

## ISSUE-0023: Near-ground atmosphere lost altitude precision and disagreed on ground hits

- Status: `fixed in code; visual acceptance pending`
- Phase: M3 spherical geometry / Sky-View composition
- Symptom: Earth-radius views showed horizontal dawn/dusk color layers, a
  black sea-level horizon edge, and at roughly 0.5 m altitude repeated broken
  black rings and speckles. The rings were already present in `Camera T`, so
  they preceded radiance integration and display mapping.
- Root cause: view-dependent kernels received `camera_radius` as f32. Around
  6.36 million metres its ULP is approximately 0.5 m, so adding sub-metre
  clearance to the radius destroyed the altitude before any integration began.
  Several paths then evaluated `radius^2 - bottom_radius^2` or an equivalent
  discriminant, magnifying cancellation at grazing incidence. Finally,
  atmosphere interval construction and Transmittance-LUT sampling made
  independent ground-hit decisions; one disagreement was converted into an
  exact black transmittance sample.

### Fix

CPU code now subtracts the two float64 radii once and passes camera altitude as
an independent f32 scalar. GPU shell clearance and horizon coordinates use
`h(2R+h)`. Camera rays use a cancellation-resistant quadratic and produce one
authoritative interval/ground result; Camera-T and Sky-view no longer recover
height from an Earth-scale radius or repeat the unstable squared-radius test.
Transmittance-LUT coordinates likewise use altitude and factored shell height.
Surface lighting and camera-to-surface reconstruction derive endpoint altitude
from a stable radial-delta expression.

The low-resolution Sky-View cache remains appropriate for smooth regions, but
is not sampled as the final authority around the projected horizon. A narrow
smooth band now uses per-pixel altitude-warped integration with local
terminator refinement. New controls are `sky_horizon_direct_steps` and
`sky_horizon_direct_width_cosine`; the Earth preset spends the higher count
only in that band.

### Verification

Configuration parsing and source compilation pass. The focused 64x48
Earth-radius CPU render regression JIT-compiles both raster paths and every
atmosphere diagnostic, checks finite/bounded outputs, exercises live/frozen LUT
updates and passes. Interactive CUDA checks at 0.1-10 m altitude and the
reported dawn/dusk views remain user-run.

### Remaining limitation

The solar-shadow-cylinder feature locator still operates on planet-centred f32
vectors. It is used only to allocate extra quadrature, not to make the physical
visibility decision, and the new stable interval prevents it from creating
black Camera-T samples. If CUDA visual acceptance finds only a residual smooth
terminator bias, that locator should next receive an altitude-relative form;
it must not be addressed by restoring duplicated ground tests or full-screen
high-count marching.
