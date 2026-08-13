# AIO3-v1 退化感知 DCNv4 U-Net 消融实验计划

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-08-13`
- Verification Status: `UNVERIFIED`
- Version Label: `aio3_dcnv4_ablation_plan_v1`
- Experiment Type: `training + mechanism validation`

## 1. 文档状态与适用范围

本文档规定退化感知 DCNv4 U-Net 的消融实验矩阵、训练顺序、评测口径和结果判定规则。所有正式结果必须遵循 `AIO3-v1`，具体训练与 runner 规范以以下文件为准：

- [`AIO3_UFORMER_TRAINING.md`](../../uformer4test/docs/AIO3_UFORMER_TRAINING.md)
- [`AIO3_RUNNER_PROVENANCE.md`](../../uformer4test/docs/AIO3_RUNNER_PROVENANCE.md)
- [`AIO3_MODEL_COMPARISON_STANDARD.md`](AIO3_MODEL_COMPARISON_STANDARD.md)
- [`AIO3_TRAINING_EVALUATION_PROTOCOL.md`](AIO3_TRAINING_EVALUATION_PROTOCOL.md)

本文档只定义实验，不代表相关变体已经在代码中实现。当前 runner 只注册了 `baseline` 和 `degradation-aware` 两个模型变体；开始消融前，必须先实现第 8 节所述的消融注册表、参数量冻结和 checkpoint 元数据校验。

已有原始 DCNv4 U-Net 和完整退化感知模型的 seed 3407 结果属于历史结果。由于其测试指标在制定本计划前已经可见，二者可作为参考锚点，但不能将整套实验描述为严格的事前预注册。所有新增变体必须在首次正式测试前冻结结构、planned contrasts 和判定标准。

## 2. 实验目标与研究问题

### 2.1 总体目标

判断 DACG 启发的退化感知机制是否适合卷积块和 DCNv4 组成的 U-Net，并把完整模型的性能提升分解为可验证的模块贡献。

### 2.2 研究问题

1. Adaptive Gated Fusion（AGF）、Prompt Conditioning 和 Context-Gated Dual-Domain Modulation（CGDM）分别贡献多少？
2. 三个模块组之间是否存在协同或负交互？
3. 当前 prompt-conditioned DCNv4 是否真正使用退化上下文，还是普通卷积 FiLM 已经解释了主要收益？
4. 完整模型的提升是否主要来自新增参数、FFT bottleneck 或 decoder 拓扑变化？
5. DAM 表征是否编码了退化类别和强度，而不是只编码图像内容或数据集身份？

### 2.3 核心假设

- `H1`：AGF 能通过过滤 encoder skip 中的退化响应提高统一恢复性能。
- `H2`：Prompt Conditioning 能够提升至少两类任务，且 prompt 干预会造成可复现的性能变化。
- `H3`：CGDM 的收益不能完全由等参数纯空间模块或静态频率门控解释。
- `H4`：完整模型的提升不是单一模块容量增加导致，而包含两个或更多模块之间的互补作用。

## 3. AIO3-v1 冻结规范

所有消融只允许改变模型结构。公共数据、训练循环、优化目标、指标和 checkpoint 选择逻辑必须保持冻结。

| 项目 | 固定配置 |
|---|---|
| 训练 patch | `128 × 128` |
| 有效 batch | 12：denoise、derain、dehaze 各4张 |
| 初始化 | 随机初始化，不加载预训练权重 |
| 精度 | BF16 autocast；DCNv4 和 FFT 局部 FP32 |
| 损失 | mean pixel L1 |
| 优化器 | AdamW，LR `2e-4`，betas `[0.9, 0.999]`，weight decay `1e-4` |
| 调度器 | 2000-step linear warmup + cosine，最低 LR `1e-6` |
| 梯度裁剪 | global norm `1.0` |
| smoke | 100 optimizer steps |
| pilot | 5,000 optimizer steps，从随机初始化开始 |
| formal | 200,000 optimizer steps，从随机初始化开始 |
| 正式验证 | 每5000 step，420张原始分辨率图，batch size 1 |
| checkpoint 选择 | 只使用未平滑的 `val/macro/psnr` |
| 正式测试 | 804张原始分辨率图，无resize、无TTA、无tile |
| 主指标 | AIO3 task-macro PSNR |
| 辅助指标 | macro SSIM、逐任务 PSNR/SSIM、参数量、FLOPs、显存和速度 |

禁止针对某个变体单独改变 loss、学习率、训练步数、patch、有效 batch、采样比例、增强、EMA、TTA 或预训练策略。显存不足时只允许使用梯度累积，并保持每个 optimizer step 的有效 batch 为4/4/4。

### 3.1 冻结 manifest

| 文件 | SHA256 |
|---|---|
| `train.jsonl` | `bd153a3b211957184de7b6171d6bc06a48f321b1c571906604d869b1aa19ca7e` |
| `val.jsonl` | `9c66c4c74a0279858ecab33df998b8eb55d6df021d2e59bbd1c253830ab3f50b` |
| `test.jsonl` | `7d80fd0af7aeaac2b6e901e20e71a744d7d705f98641f54913aa278e12c2b63a` |
| `data_audit.json` | `2959e402ecdb76172b9fe9bba3fae13c090348379dcd33898992abdc198e06b8` |
| `visual_samples.json` | `62e9f6e761e3db2c23895958f3707414a59baac30840919044fe6bf848ff628b` |

任一哈希不一致时停止实验，不得重新生成或覆盖 manifest 后继续声称使用 `AIO3-v1`。

## 4. 核心 2×2×2 因子消融

### 4.1 因子定义

| 因子 | 名称 | 启用时包含的结构 |
|---|---|---|
| `A` | AGF | 三个 `AdaptiveGatedFusion`，过滤 encoder skip 后再与 decoder feature 融合 |
| `P` | Prompt Conditioning | DAM stage prompts、普通卷积 FiLM、DCNv4 input affine 和 DCNv4 output gate |
| `G` | CGDM | DAM global context 和 bottleneck 双域调制 |

当 `P=1` 和 `G=1` 时，共享同一个 DAM。只有 `G=1` 时，DAM 仅生成和使用 global context，stage prompts 不进入 encoder、decoder 或 residual head。只有 `P=1` 时，DAM 只生成 stage prompts，CGDM 路径不存在。

### 4.2 八组核心模型

| ID | A | P | G | 配置 | 主要问题 |
|---|:---:|:---:|:---:|---|---|
| `APG-000` | × | × | × | 原始 DCNv4 U-Net | 冻结基线 |
| `APG-100` | ✓ | × | × | 仅 AGF | skip 门控是否有效 |
| `APG-010` | × | ✓ | × | 仅完整 Prompt Conditioning | 当前 DCNv4 适配是否有效 |
| `APG-001` | × | × | ✓ | 仅 DAM global context + CGDM | 双域模块是否有效 |
| `APG-110` | ✓ | ✓ | × | AGF + Prompt Conditioning | 二者是否互补 |
| `APG-101` | ✓ | × | ✓ | AGF + CGDM | 不依赖 DCNv4 条件化时的效果 |
| `APG-011` | × | ✓ | ✓ | Prompt Conditioning + CGDM | 去掉 AGF 后的退化感知路径 |
| `APG-111` | ✓ | ✓ | ✓ | 当前完整模型 | 最终组合及三因素协同 |

### 4.3 因子正交要求

每个 ID 只能改变表中指定因子，不能因为复用类实现而夹带其他拓扑变化：

- `A=0`：必须使用原始 concat skip；
- `A=1`：必须使用 AGF；
- `P=0`：普通 DCNv4 block、普通卷积 decoder、普通 residual head 均不能接收 prompt；
- `P=1`：启用当前 stage prompt conditioning；
- `G=0`：bottleneck 输出直接进入 decoder；
- `G=1`：bottleneck 输出经过 CGDM；
- `APG-010` 必须是“原始 concat skip + prompt-conditioned features”，不能意外采用 AGF 版本 decoder 的通道压缩拓扑。

## 5. DCNv4 条件化机制消融

这一组关闭 AGF 和 CGDM，只检查从 CAGA 思路适配到 DCNv4 的具体路径。所有变体使用同一个 DAM 生成 stage prompts。

| ID | 普通卷积 FiLM | DCNv4 input affine | DCNv4 output gate | 目的 |
|---|:---:|:---:|:---:|---|
| `P-FILM` | ✓ | × | × | 判断普通卷积条件化能否解释主要收益 |
| `P-DIN` | × | ✓ | × | 检查 prompt 间接影响 DCNv4 offset/weight 预测是否有效 |
| `P-DOUT` | × | × | ✓ | 检查只门控 DCNv4 residual response 是否有效 |
| `P-DIO` | × | ✓ | ✓ | 只条件化 DCNv4，不调制普通卷积 |
| `APG-010` | ✓ | ✓ | ✓ | 当前完整 Prompt Conditioning 路径 |

planned contrasts：

- `P-DIN − APG-000`：DCNv4 输入条件化贡献；
- `P-DOUT − APG-000`：DCNv4 输出门控贡献；
- `P-DIO − P-FILM`：DCNv4 专属条件化是否优于通用卷积 FiLM；
- `APG-010 − P-DIO`：普通卷积层接受 prompt 的额外贡献；
- `APG-010 − P-FILM`：DCNv4 专属路径在完整 conditioning 中的净贡献。

如果 `P-FILM` 与 `APG-010` 基本相同，应将结论限定为“全局退化上下文对卷积特征调制有效”，不能声称当前 DCNv4 专属条件化获得了独立支持。

## 6. 拓扑和参数量混杂控制

### 6.1 AGF 拓扑对照

AGF 同时引入 gate 和 `1×1` 融合投影。为隔离真正的 gate 贡献，增加以下对照：

| ID | 配置 | 主要比较 |
|---|---|---|
| `SKIP-PROJ-000` | `concat(encoder, decoder) → 1×1 + GELU → decoder block`，无门控 | `APG-100 − SKIP-PROJ-000` |
| `SKIP-PROJ-011` | P+G 完整启用，skip 使用相同投影但不使用 AGF gate | `APG-111 − SKIP-PROJ-011` |

只有 AGF 优于对应 `SKIP-PROJ` 时，提升才可以归因于 skip filtering，而不是 decoder 通道数和投影拓扑变化。

### 6.2 CGDM 对照

| ID | 配置 | 主要问题 |
|---|---|---|
| `CGDM-STATIC` | 保留 FFT、frequency mixer 和 fusion，但使用样本无关的可学习频率 gate | 输入相关 global context 是否必要 |
| `CGDM-SPATIAL` | 保留 DAM/context mapper，以参数量接近的纯空间分支替换 FFT 分支 | 频域设计是否优于单纯增加 bottleneck 容量 |

`CGDM-SPATIAL` 的 trainable parameter count 应控制在正式 CGDM 的 ±1%，同时报告 FLOPs、峰值显存和吞吐量。

planned contrasts：

- `APG-001 − CGDM-STATIC`：退化自适应频率门控贡献；
- `APG-001 − CGDM-SPATIAL`：频域处理贡献；
- `CGDM-SPATIAL − APG-000`：额外 bottleneck 容量贡献。

## 7. 无需重新训练的 prompt 干预实验

对 `APG-010`、`APG-011` 和 `APG-111` 的冻结正式 checkpoint，在验证集上进行以下干预。干预结果不参与 checkpoint 选择，也不使用 test 集。

| 干预 | 操作 | 解释目标 |
|---|---|---|
| `correct` | 使用输入图像自身的 prompt | 原始验证指标 |
| `neutral` | FiLM scale/shift 强制为0，DCNv4 output gate强制为1 | 移除 prompt 的实际作用 |
| `same-task-swap` | 使用同任务另一图的 prompt | 区分退化信息与图像内容 |
| `cross-task-swap` | 使用其他任务的 prompt，按固定 sample ID 循环映射 | 检查任务相关退化表征 |
| `severity-swap` | 同一 WED scene 的 σ15 与 σ50 prompt 对换 | 检查退化强度感知 |

验证 batch size 固定为1，因此不能依赖临时 batch shuffle。应先为420个验证条件离线提取 prompt，再按冻结的 sample ID 映射注入。所有映射和随机种子必须写入诊断输出。

解释规则：

- `neutral` 几乎不下降：conditioning 路径可能没有被模型实际使用；
- same-task swap 基本不下降、cross-task swap 明显下降：prompt 更可能编码退化类别；
- σ15/σ50 对换产生方向一致的下降：支持强度感知；
- 所有 swap 都明显下降：prompt 可能混入大量图像内容；
- 所有 swap 都不下降：不能支持“退化感知”结论。

## 8. 实现和接入要求

### 8.1 消融注册表

在开始训练前实现统一的 ablation registry。每个变体必须冻结并记录：

- 唯一 `variant_id`；
- A/P/G 和子模块开关；
- skip fusion 类型；
- bottleneck modulation 类型；
- DCNv4 conditioning 类型；
- architecture version；
- expected trainable parameters；
- autocast/FP32 边界；
- checkpoint metadata；
- W&B tags 和输出目录。

建议所有变体由同一个可配置模型类或少量正交组件构建，避免复制16份模型代码。公共 runner 不得因变体改变 sampler、loss、optimizer、schedule、metric、validation 或 evaluation。

### 8.2 参数量和结构测试

每个变体至少需要以下自动化测试：

1. 参数量与 registry 常量一致；
2. 输入输出尺寸完全一致并支持任意原始分辨率 padding/crop；
3. 初始化时完整模型满足 `output == input`；
4. 禁用模块确实不存在或不参与 forward；
5. 启用模块的梯度 finite 且非零；
6. BF16 网络、FP32 DCNv4/FFT 路径 forward/backward 通过；
7. checkpoint round-trip 后固定输入输出一致；
8. baseline 和历史 full model 在无数学改动时保持结构回归一致；
9. 公共 runner provenance/hash 测试继续通过。

### 8.3 分支与冻结

建议使用单独分支：

```text
aio3-dcnv4-ablation-v1
```

正式训练前必须提交一个干净的冻结 commit。任何影响 forward 数学行为的修改都需要新的 commit，并从随机初始化重新训练受影响的变体。

## 9. 训练执行流程

### 9.1 固定服务器路径

```bash
export PROJECT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation
export MODEL_ROOT="${PROJECT_ROOT}/all-in-one-model"
export REPO_ROOT="${MODEL_ROOT}/DCNv4"
export OUTPUT_ROOT="${PROJECT_ROOT}/outputs/AIO3/aio3-v1"
export MANIFEST_DIR="${OUTPUT_ROOT}/manifests"

cd "${REPO_ROOT}"
```

### 9.2 开始每一批实验前验证 manifest

```bash
sha256sum \
  "${MANIFEST_DIR}/train.jsonl" \
  "${MANIFEST_DIR}/val.jsonl" \
  "${MANIFEST_DIR}/test.jsonl" \
  "${MANIFEST_DIR}/data_audit.json" \
  "${MANIFEST_DIR}/visual_samples.json"
```

### 9.3 每个可报告变体的完整验收顺序

1. 运行完整 pytest 和真实 CUDA 模型集成测试；
2. 运行100-step smoke；
3. 创建独立 smoke，在 step 50 暂停并从同一 `latest.pth` 恢复到 step 100；
4. 从随机初始化运行5,000-step pilot；
5. 人工检查14个固定验证样本；
6. 废弃 pilot checkpoint；
7. 从随机初始化运行200,000-step formal；
8. 只按验证 macro PSNR 选择 `best_macro_psnr.pth`；
9. 所有结构决策冻结后，正式 test 只运行一次。

pilot 只用于排查 OOM、NaN/Inf、无梯度、恢复失败、严重边缘伪影或 gate 坍缩。不得因 pilot PSNR 较低而为单个变体修改学习率、损失、训练步数或数据配置。

### 9.4 通用命令模板

以下命令要求第 8 节的变体 registry 已经接入 `--model-variant`：

```bash
export VARIANT=APG-100
export SEED=3407

RUN_NAME="dcnv4-ablation-${VARIANT}-smoke-seed${SEED}-$(date -u +%Y%m%d-%H%M%S)"
python -m aio3_runner.train \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-variant "${VARIANT}" \
  --run-kind smoke \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

正式训练只把 `--run-kind smoke` 改为 `--run-kind formal`，并使用新的 `RUN_NAME`。不得从 smoke 或 pilot checkpoint 继续。

中断恢复只能使用同一 run 的 `latest.pth`：

```bash
python -m aio3_runner.train \
  --resume "${RUN_DIR}/checkpoints/latest.pth"
```

正式训练完成且 `run_state.json` 为 `completed/200000` 后才能运行：

```bash
python -m aio3_runner.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/best_macro_psnr.pth" \
  --num-workers 4
```

## 10. 实验分阶段与算力预算

### 10.1 Phase I：实现与预检

- 实现16个唯一变体：8个核心因子模型、4个 conditioning 机制模型、2个 AGF 拓扑对照和2个 CGDM 对照；
- 冻结参数量、checkpoint metadata 和输出目录；
- 所有变体完成单元测试、CUDA 测试、smoke、resume smoke 和 pilot。

### 10.2 Phase II：核心单种子实验

对八组 `APG-*` 运行 seed 3407 的200k formal。这一阶段给出完整的 2×2×2 因子表。

如果已有 `APG-000` 和 `APG-111` 满足以下全部条件，可以复用其 seed 3407 正式结果：

- manifest 哈希完全一致；
- 公共 runner 行为和指标实现一致；
- 模型 forward 未因重构发生数学变化；
- 原 run 为干净 commit 下的 `completed/200000`；
- best checkpoint 和正式 test 输出已经冻结。

若重构改变任何 forward 路径，必须重跑对应模型。

### 10.3 Phase III：机制和混杂对照

使用 seed 3407 运行：

- `P-FILM`、`P-DIN`、`P-DOUT`、`P-DIO`；
- `SKIP-PROJ-000`、`SKIP-PROJ-011`；
- `CGDM-STATIC`、`CGDM-SPATIAL`。

### 10.4 Phase IV：多种子确认

论文级核心表使用：

```text
seed = 3407, 3408, 3409
```

最低完成八组 `APG-*` 的三种子结果，即24个 formal run。8个机制/混杂对照至少完成 seed 3407；如果资源允许，再扩展到三种子。

| 证据等级 | formal run 数量 | 用途 |
|---|---:|---|
| 探索性核心 | 8组 × 1 seed = 8 | 初步模块消融，不足以估计训练方差 |
| 推荐核心 | 8组 × 3 seeds = 24 | 论文核心消融表 |
| 推荐核心+控制 | 24 + 8个控制的seed3407 = 32 | 骨干兼容性和混杂分析 |
| 最强验证 | 16组 × 3 seeds = 48 | 全部模块和控制的跨种子确认 |

合法复用已有 seed 3407 baseline/full 时，可以减少对应的新训练数量，但必须在结果表中记录其 commit、run ID 和复用理由。

## 11. 特征与门控诊断

诊断脚本独立读取冻结 checkpoint 和验证 manifest，不修改公共 runner，也不参与模型选择。

### 11.1 DAM 表征

- global context 和四级 prompt 的均值、标准差与范数；
- 任务 linear probe；
- silhouette score；
- 同一 WED scene 下 σ15/25/50 的嵌入距离和强度单调性；
- 任务标签和数据集标签分别分析，避免把数据集内容差异误判为退化感知。

### 11.2 AGF

每层、每任务报告：

- gate mean/std；
- gate `<0.05` 和 `>0.95` 的饱和比例；
- encoder feature 过滤前后范数；
- 三任务 gate 分布差异。

### 11.3 CGDM

- frequency gate mean/std；
- real/imag gate 分布；
- 三任务差异；
- gate 是否坍缩为输入无关常数。

### 11.4 DCNv4

通过真实 CUDA DCNv4 的 `offset_mask` hook 报告：

- offset 平均模长、最大值和分位数；
- aggregation weight 的 L1/L2；
- 正/负权重比例；
- 按绝对值归一化后的诊断性权重熵；
- correct、neutral 和 cross-task prompt 下上述统计的变化。

诊断性权重熵不能称为 DCNv4 attention temperature，因为 DCNv4 聚合权重没有 softmax 概率解释。

## 12. 输出、监控与成功标准

### 12.1 每个 run 的必需输出

| 输出 | 格式 | 成功标准 |
|---|---|---|
| `config.yaml` | YAML | 包含完整 variant registry、seed、manifest hash 和 commit |
| `environment.json` | JSON | 记录 Python、PyTorch、CUDA、GPU 和 DCNv4 环境 |
| `run_state.json` | JSON | formal 为 `completed` 且 step=200000 |
| `train_metrics.jsonl` | JSONL | loss、grad norm、速度、显存和4/4/4采样均 finite |
| `validation_metrics.jsonl` | JSONL | 每5000 step存在完整420图验证结果 |
| `checkpoints/latest.pth` | PyTorch | 可回读并用于精确恢复 |
| `checkpoints/best_macro_psnr.pth` | PyTorch | 仅由 `val/macro/psnr` 选择 |
| `test/metrics.json` | JSON | 冻结测试完成并包含全部任务和macro指标 |
| `test/per_image_metrics.csv` | CSV | 805行，含表头 |
| `test/predictions/` | PNG | 804张预测 |
| `test/gallery/` | PNG | 70张固定gallery图 |

### 12.2 监控

训练过程中持续检查：

- 进程存活和 `run_state.json`；
- 最近一条 `train_metrics.jsonl`；
- W&B run ID 和上传错误日志；
- NaN/Inf、梯度范数、吞吐量和峰值显存；
- `train/samples_denoise`、`derain`、`dehaze` 始终对应最近50步的 `200/200/200`；
- 每10,000 step 的14个固定视觉样本。

终端意外关闭后，先检查进程是否仍然存在，再决定是否使用同一 run 的 `latest.pth` 恢复；禁止新建 run 接着写旧实验结果。

## 13. 指标与统计分析

### 13.1 每个 seed 的报告项

- BSD68 σ15、σ25、σ50 和 mean PSNR/SSIM；
- Rain100L PSNR/SSIM；
- SOTS Outdoor PSNR/SSIM；
- AIO3 task-macro PSNR/SSIM；
- trainable parameters、FLOPs、峰值显存、吞吐量和平均推理时间；
- best validation step、best validation macro PSNR/SSIM；
- run commit、checkpoint SHA256 和 W&B URL。

### 13.2 多种子汇总

三种子结果报告 `mean ± sample standard deviation`，同时保留每个 seed 的原始数值。不能只报告最佳 seed。

### 13.3 因子分析

对八组核心模型按 seed 计算：

```text
metric ~ A * P * G + seed
```

报告 A/P/G 主效应、三组双因素交互和 A×P×G 三因素交互。主效应定义为该因子启用时四组结果均值减去关闭时四组结果均值；交互使用 difference-in-differences。

由于 `G=1` 需要 DAM global context，这些结果应称为“模块组效应”，不能声称是严格参数匹配的单层因果贡献。

### 13.4 配对 bootstrap

可以对冻结 test 的 per-image metrics 做辅助性配对 bootstrap：

- BSD68 以68个 scene 为重采样单位，并绑定同一 scene 的三个 sigma；
- Rain100L 以图像对为单位；
- SOTS Outdoor 以 scene 为单位；
- 每次先计算三个任务指标，再按任务等权计算 macro；
- 建议10,000次 bootstrap，报告95% percentile CI。

图像 bootstrap 只反映给定训练 checkpoint 下的样本不确定性，不能代替跨训练 seed 方差。

## 14. 预先冻结的判定规则

1. **模块有效**：三种子平均 macro PSNR 至少提高约0.10 dB，且至少2/3 seeds 同方向。
2. **统一恢复改善**：任一任务的三种子平均回退不超过0.10 dB；超过时必须报告负迁移，不能只报告macro。
3. **Prompt Conditioning 得到支持**：P 对参数/拓扑对照有稳定提升，并且 neutral 或 cross-task prompt 干预造成可复现下降。
4. **AGF 得到支持**：必须优于相应 `SKIP-PROJ`，而不只是优于原始 concat。
5. **CGDM 的退化自适应得到支持**：必须优于 `CGDM-STATIC`。
6. **CGDM 的频域设计得到支持**：必须优于参数匹配的 `CGDM-SPATIAL`。
7. **退化表征得到支持**：context/prompt 没有坍缩成常数，并在控制图像内容后表现出任务或强度相关差异。
8. **协同作用得到支持**：交互项在多数 seed 同方向，不能只由一个异常 seed 或单任务极端提升驱动。

0.10 dB 是本计划用于筛查实践意义的工作阈值，不是统计显著性阈值，也不替代跨种子方差和置信区间。

## 15. 允许的结论边界

### 15.1 如果只有 AGF 和 CGDM 有效

应表述为：

> DACG 启发的跳接过滤与双域增强适合卷积 U-Net，但当前 CAGA 到 DCNv4 的条件化映射尚未获得充分支持。

### 15.2 如果 P-FILM 与完整 P 相近

应表述为：

> 输入推断的退化上下文对卷积特征调制有效，但尚不能证明 DCNv4 专属采样/响应条件化提供了独立收益。

### 15.3 如果 DCNv4 专属路径和 prompt 干预均有效

可以表述为：

> DCNv4 U-Net 能够利用输入推断的退化上下文，自适应调节其特征采样与响应，并改善统一图像恢复。

即使得到第三种结果，也应将方法称为 `DACG-inspired degradation-conditioned DCNv4`，而不是 CAGA 的数学等价实现。

## 16. 执行前检查清单

- [ ] 16个变体均在统一 registry 中注册；
- [ ] 每个变体的参数量、结构开关和 metadata 测试通过；
- [ ] 公共 runner provenance/hash 测试通过；
- [ ] 五个 manifest SHA256 完全一致；
- [ ] Git 工作区干净且正式 commit 已冻结；
- [ ] 真实 DCNv4 CUDA forward/backward 和 BF16/FP32 边界通过；
- [ ] 每个变体完成100-step smoke；
- [ ] 每个变体完成独立 pause/resume smoke；
- [ ] 每个变体完成从头开始的5000-step pilot；
- [ ] 14个固定样本无明显异常；
- [ ] formal 从随机初始化开始，不恢复 pilot；
- [ ] checkpoint 只按 validation macro PSNR 选择；
- [ ] 新增变体和 planned contrasts 在首次 test 前全部冻结；
- [ ] test 只运行一次且不用于后续结构选择；
- [ ] 结果表同时报告逐任务指标、macro、参数量、速度和跨种子方差。
