# 退化感知 DCNv4 U-Net：溯源、诊断与设计

## 结论

现有 `DCNv4RestorationUNet` 已经把旧分解网络不适合 AIO 的双头输出改成了
单一有符号残差，但它仍是一个明确的 **degradation-agnostic baseline**。DCNv4
只能依据当前局部特征预测采样偏移和聚合权重；网络没有从输入中提取全局退化
表征，也没有利用该表征调节各层计算。三种任务共享同一组静态卷积、同一组跳连
和同一瓶颈，因而容易发生去噪所需的平滑、去雨所需的方向结构抑制、去雾所需的
低频/对比度校正之间的负迁移。

为保留严格的 AIO3-v1 对照，原 baseline 未被改写。本次新增
`DegradationAwareDCNv4RestorationUNet`，将 DACG-IR 的退化感知、门控跳连和
双域调制迁移到现有 CNN + DCNv4 主干。

## 1. 代码演化与原始任务痕迹

当前 Git 历史可以还原出以下路径：

1. 初始提交 `ae9d055` 的 `GeneralDecompositionNet` 输出两个分支：非负
   `pattern` 与 `[0,1]` 范围的 `background`，并通过正交损失分离两类特征。
2. 初始数据接口把背景真值命名为 `lol_gt`，把退化图案真值命名为
   `flare_gt`。这些字段以及 `pattern + background ≈ input` 的加性约束，清楚地
   保留了低照度/炫光分解任务的来源痕迹。
3. 提交 `322a299` 曾加入仅面向雨纹的 `RainAttentionGate`，随后在
   `a85501f`/`0602d21` 被回退；这也说明当时的门控不是统一退化建模。
4. 提交 `3bf6163` 用 DCNv4 替换方向卷积，但 DCNv4 仍只是通用特征算子，并未
   获得图像级退化上下文。
5. 提交 `66f21d7` 新增 AIO baseline：删除分解双头，改为零初始化的有符号 RGB
   残差头，并将 BatchNorm 改为 GroupNorm。这解决了输出约束与混合任务统计问题，
   但编码器、瓶颈和普通拼接跳连仍继承自原始 U-Net。

因此，“输出形式已适配恢复任务”不等于“主干已具备 AIO 退化感知能力”。

## 2. 当前 baseline 的关键缺口

| 位置 | 当前行为 | 对 AIO 的风险 |
|---|---|---|
| 输入侧 | 直接进入共享卷积编码器 | 没有退化类型、强度或组合状态的隐式提示 |
| DCNv4 | 偏移/权重只由当前层局部特征产生 | 能内容自适应，但没有图像级退化条件 |
| 跳连 | encoder 与 decoder 特征直接拼接 | 浅层噪声、雨纹和雾化响应不经筛选地回流到重建端 |
| 瓶颈 | 四层空域膨胀卷积 | 对去雾等低频全局退化以及雨纹频谱缺少显式建模 |
| 多任务路由 | 所有样本经过相同静态路径 | 容易产生任务梯度冲突和折中解 |

## 3. 从 DACG-IR 到当前 DCNv4 主干的映射

参考论文：Lei He 等，*Degradation-Aware Adaptive Context Gating for Unified
Image Restoration*，arXiv:2605.01236。参考实现：
`https://github.com/HlHomes/DACG-IR-code`。
截至本设计记录，引用的是 2026-05-02 的 arXiv v1 预印本；论文页面注明已投稿
IEEE TIP，文中增益应视为作者报告结果，不能代替本项目的独立复现。

| DACG-IR 组件 | 本实现 | 适配理由 |
|---|---|---|
| DAM | `DegradationAwareContext` | 3/5/7 深度卷积分支、空间门控、mean/std 双统计池化，生成四层 prompt 与全局 context；无需退化标签 |
| CAGA | `ContextGatedDCNv4FeatureBlock` | DCNv4 没有可直接调节的 softmax temperature，因此改为 prompt-Film 调节 DCNv4 输入，使 offset/mask 预测获得退化条件，再对 DCNv4 残差响应做通道门控 |
| AGF | `AdaptiveGatedFusion` | 用 encoder/decoder 联合产生空间门与通道门，先过滤 encoder skip，再融合到 decoder |
| CGDM | `ContextGatedDualDomainModulation` | 仅在最低分辨率瓶颈执行 FFT；全局 context 产生频率通道门，同时保留空域深度卷积分支 |
| 分层 prompt | encoder DCNv4、decoder feature modulation、residual head | 让同一图像的退化上下文贯穿编码、融合和输出，而不是只在 decoder 末端注入 |

### 与官方代码的有意差异

- 论文公式中的 DAM 空间门是深度 3×3 卷积，而官方 `model.py` 当前实现为
  1×1 卷积；本实现采用论文公式的深度 3×3 版本。
- CAGA 的 temperature 只适用于显式注意力。这里没有伪造“DCNv4 temperature”，
  而是调节 DCNv4 输入与输出门；其效果需独立消融验证。
- prompt 调制层零初始化，初始缩放为 1；RGB residual head 继续零初始化。因此新
  网络在初始化时仍严格满足 `output == input`。
- FFT、频率混合和 context-to-frequency gate 在 FP32 中运行；其余网络继续兼容
  AIO3 的 BF16 autocast。四个 DCNv4 算子仍沿用已有 FP32 隔离。
- 训练不使用任务标签。manifest 中的 `denoise/derain/dehaze` 标签仅用于平衡采样、
  分任务指标和后续表征诊断，不进入网络前向。

## 4. 规模、兼容性与运行方式

| 模型 | 参数量 | DCNv4 块数 | checkpoint |
|---|---:|---:|---|
| 原 baseline | 29,924,411 | 4 | architecture version 3 |
| 退化感知变体 | 34,852,539 | 4 | architecture version 4 |

新增 4,928,128 个参数（约 +16.47%），主要来自瓶颈频率混合、DAM 和三个 AGF；
DCNv4 调用次数没有增加。两种 checkpoint 被严格区分，不能误加载。

新训练显式选择变体：

```bash
python -m aio3_runner.train \
  --manifest-dir /path/to/aio3-v1/manifests \
  --output-root /path/to/outputs/AIO3/aio3-v1 \
  --model-variant degradation-aware \
  --run-kind smoke \
  --run-name dcnv4-dacg-smoke-seed3407 \
  --seed 3407 \
  --wandb-mode offline
```

输出写入独立的 `degradation_aware_dcnv4_unet/<run_name>`，原
`dcnv4_unet/<run_name>` baseline 目录不受影响。省略 `--model-variant` 时仍使用
原 baseline。

## 5. 验证状态与实验计划

当前已经验证的是实现正确性，不是恢复指标提升：

- 任意尺寸前向与裁剪正确；
- 两个模型初始化均为逐像素精确 identity；
- DAM prompt 尺寸与四层通道一致，并随输入变化；
- 重建损失能反传至退化上下文 stem；
- 三个 AGF、一个 CGDM、四个 context-gated DCNv4 块均存在；
- FP32 FFT/BF16 外层路径可微；
- baseline 回归测试保持通过；
- 参数量与 runner 冻结配置一致。

训练服务器编译真实 DCNv4 扩展后，还必须执行 CUDA 集成测试；本地 FakeDCNv4
结构测试不能替代真实 kernel 验证：

```bash
python scripts/test_dcnv4_restoration.py \
  --model-variant degradation-aware \
  --batch-size 1 --height 128 --width 128
```

性能结论必须通过同一 AIO3-v1 manifest、采样器、L1、优化器、训练步数和选择规则
获得，不能直接借用论文中的数值。推荐按以下顺序做消融：

1. `B0`：原 `DCNv4RestorationUNet`。
2. `B1`：只替换 AGF，验证浅层退化噪声传播假设。
3. `B2`：DAM + prompt-conditioned DCNv4，不启用 CGDM。
4. `B3`：DAM + CGDM，不替换 skip。
5. `B4`：完整退化感知模型。

主要判据为 task-macro PSNR/SSIM，同时逐任务报告，避免平均值掩盖某一任务退化。
建议增加三个不参与模型选择的诊断：

- global context 的 t-SNE/UMAP 与 silhouette score（按任务及噪声强度着色）；
- 三任务梯度余弦相似度，检查负迁移是否下降；
- AGF mask 与频率 gate 的均值、方差和任务间差异，确认门控没有塌缩为常数。

只有 `B4` 在至少两个种子上提高 task-macro 指标、且任一任务没有不可接受的明显
回退时，才能把“退化感知提高 AIO 能力”从设计假设升级为实验结论。
