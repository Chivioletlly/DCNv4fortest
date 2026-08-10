# AIO-3 统一训练与评测协议（AIO3-v1）

## 1. 文档状态与适用范围

本文档冻结 AIO-3 图像恢复实验的第一版统一协议，协议标识为
`AIO3-v1`。当前执行模型为 `DCNv4RestorationUNet`，以后接入其他网络时，
除模型适配器和模型自身初始化外，不得修改本文规定的数据、采样、训练、验证、
测试和指标实现。

本协议的目标是比较不同网络在同一数据和优化条件下的恢复能力，不追求复现某一篇
论文的模型专属训练技巧。若改变数据划分、损失函数、噪声生成、采样比例或评测实现，
必须使用新的协议版本，不能继续标记为 `AIO3-v1`。

其他模型贡献者可先阅读执行摘要、接入清单和合规报告模板：
[`AIO3_MODEL_COMPARISON_STANDARD.md`](AIO3_MODEL_COMPARISON_STANDARD.md)。

当前已验证的 DCNv4 模型约束如下：

- 模型输入和输出均为 `[B, 3, H, W]` RGB 张量；
- 模型预测有符号残差，最终输出为 `degraded + signed_residual`；
- 最终 RGB 残差分支没有 ReLU、Sigmoid、Tanh 或训练时 clamp；
- 网络外层使用 BF16 autocast，四个 DCNv4 算子始终隔离 autocast 并以 FP32
  执行；
- 模型参数保持 FP32，禁止直接调用 `model.bfloat16()`；
- 模型内部负责 padding，并将输出裁剪回原始输入尺寸。

## 2. 固定目录结构

服务器项目根目录固定为：

```text
/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation
```

目录职责如下：

```text
Unet4Degradation/
├── data/
│   └── AIO3/                         # 原始数据，只读
│       ├── BSD400/
│       ├── BSD68/
│       ├── WED/
│       ├── RainTrainL/
│       ├── Rain100L/
│       ├── OTS/
│       └── SOTS/
├── all-in-one-model/
│   ├── DCNv4/                        # DCNv4恢复模型仓库
│   │   └── aio3_runner/              # 当前公共数据、训练和评测包
│   └── <future_model>/               # 未来模型与DCNv4同级存放
└── outputs/
    └── AIO3/
        └── aio3-v1/
            ├── manifests/            # 共享冻结manifest和审计结果
            └── <model_name>/
                └── <run_name>/
```

固定路径变量为：

```text
DATA_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3
MODEL_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model
OUTPUT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1
```

原始数据目录视为只读。manifest、checkpoint、预测图和 W&B 本地文件均不得写入
`DATA_ROOT` 或模型源码目录。

## 3. 已知数据状态

当前服务器数据统计如下。有效图像统计忽略隐藏文件和所有
`.ipynb_checkpoints` 目录；被有效图像规则排除的普通/隐藏文件会在审计报告中单列。

| 数据集 | 当前有效文件状态 | 用途 |
|---|---:|---|
| BSD400 | 400 张图像 | 去噪训练清晰图 |
| WED | `gt` 4744 张，`noisy` 4744 张 | 去噪训练/验证清晰图 |
| BSD68 | 68 张图像 | 去噪正式测试清晰图 |
| RainTrainL | `rain-*` 200 张，`norain-*` 200 张 | 去雨训练/验证 |
| Rain100L | `rain` 100 张，`gt` 100 张 | 去雨正式测试 |
| OTS | `clear` 2061 张，72135 张有效 haze，另有4个被排除文件 | 去雾训练/验证 |
| SOTS Outdoor | input 500 张，target 492 张，严格配对500组 | 去雾正式测试 |

`RainTrainL/rainregion-*` 不属于本协议的恢复目标，必须排除。`OTS/depth` 也不
参与 AIO-3 训练。

WED 中现有 `noisy` 图像的噪声生成方式和噪声级别没有可靠元数据。为避免不同噪声
分布混入基线，`AIO3-v1` **不使用 `WED/noisy`**，只使用 `WED/gt` 并按第 6 节
在线生成高斯噪声。

## 4. Manifest 与数据审计

训练代码不得依靠两个目录排序后的数组下标进行配对。每次首次准备数据时必须生成
JSON Lines manifest，并同时输出 `data_audit.json` 和 manifest 的 SHA256。

每条 manifest 至少包含：

```json
{
  "id": "raintrainl-001",
  "task": "derain",
  "split": "train",
  "input": "/absolute/path/rain-1.png",
  "target": "/absolute/path/norain-1.png",
  "scene_id": "1",
  "metadata": {}
}
```

对在线生成噪声的样本，`input` 指向清晰图或设为 `null`，`target` 指向清晰图，
并在 `metadata` 中保存数据源。训练时的实际 `sigma` 随每次抽样生成；固定验证和
测试 manifest 必须保存 `sigma` 与噪声种子。

配对规则：

1. RainTrainL：只匹配相同数字后缀的 `rain-N` 与 `norain-N`；
2. Rain100L：匹配 `rain/rain-N` 与 `gt/norain-N`；
3. OTS：从 haze 文件名解析 clear scene ID，再映射到唯一的 `clear/<ID>.*`；
   同一个 clear 允许对应多个 haze；
4. SOTS Outdoor：从 input 文件名解析 scene ID，再匹配对应 target；
5. 禁止按文件列表索引配对，禁止静默跳过重复 scene ID；
6. 输入与 target 解码后必须具有相同宽高，否则审计失败。

审计报告必须记录：

- 每个任务和 split 的样本数、场景数；
- 重复 ID、无法解码图像、尺寸不一致图像；
- 缺失 input 或 target 的文件列表；
- 被规则排除的文件数；
- manifest 路径、SHA256、生成时间和协议版本。

RainTrainL 和 Rain100L 的有效配对数分别必须为 200 和 100，否则停止实验。
OTS 必须确认 2061 个 clear scene 均至少拥有一个有效 haze 版本；若不满足，不得按
“1961 train + 100 val”继续训练，而应先修复数据并重新生成审计报告。
SOTS 的 500 张 input 必须全部成功配对。当前 500 张 input 映射到 492 个唯一 target
文件，说明部分 target 被多个退化 input 复用，并不表示缺少8个测试样本。正式评测包含
500个 input-target pair，同时在审计中记录492个唯一 target。

### 4.1 当前审计实现与服务器命令

第一阶段实现位于 DCNv4 仓库的 `aio3_runner` Python 包中。该包虽然随 DCNv4 仓库
分发，但 manifest 格式、划分和噪声种子逻辑不依赖 DCNv4 模型代码。

在服务器执行：

```bash
cd /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DCNv4

python -m aio3_runner.prepare_data \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3 \
  --output-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests
```

正式运行默认对所有实际使用的图像执行完整 Pillow 解码，并检查每个真实 input/target
pair 的宽高。OTS 有 72135 张有效 haze，首次审计需要一定磁盘读取时间。额外4个
非图像或隐藏文件会被排除并完整记录；不要在正式审计中使用
`--skip-image-verification`。

成功后固定生成：

```text
manifests/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── visual_samples.json
└── data_audit.json
```

默认拒绝覆盖已有输出。只有明确决定重新生成 manifest 时才允许增加 `--overwrite`，
而且重新生成后必须比较 SHA256；正式训练 run 需要复制这些文件并保存完全相同的哈希。

## 5. 固定训练/验证划分

所有划分按 scene ID 完成，禁止同一场景的不同退化版本跨越训练集和验证集。

划分排序键固定为：

```text
SHA256("aio3-v1:<task>:<scene_id>")
```

按上述哈希值从小到大排序，再选择固定数量的验证场景：

| 任务 | 训练集 | 验证集 |
|---|---|---|
| denoise | BSD400 全部 + WED 剩余图像 | WED 100 张清晰图 |
| derain | RainTrainL 剩余 180 对 | RainTrainL 20 对 |
| dehaze | OTS 剩余 1961 个 clear scene | OTS 100 个 clear scene |

去雾训练时先均匀抽取 clear scene，再从该 scene 的 haze 版本中均匀抽取一张。
去雾验证对每个验证 scene 使用一个固定 haze 版本，版本下标由
`SHA256("aio3-v1:dehaze-val:<scene_id>")` 对候选数量取模得到。这样验证集固定为
100 对，而不会因一个 clear 拥有多个 haze 版本而改变场景权重。

去噪验证对保留的 100 张 WED 清晰图分别生成 `sigma=15/25/50` 三个固定版本，
共 300 个验证条件。

## 6. 输入预处理与数据增强

### 6.1 公共处理

- 所有图像使用 Pillow 解码并显式转换为 RGB；
- 张量布局为 `[C, H, W]`，数值范围为 `[0, 1]`；
- 不对完整训练图像进行 resize；
- input 和 target 始终使用完全相同的几何变换；
- 训练 patch 固定为 `128 x 128`；
- 图像小于 patch 时先做 reflection padding；reflection 不合法时才使用
  replicate padding；
- 随机水平翻转概率为 0.5；
- 随机垂直翻转概率为 0.5；
- 随机选择 `0/90/180/270` 度旋转；
- 禁止颜色抖动、随机锐化和单独作用于 input/target 的空间增强。

### 6.2 在线高斯去噪数据

去噪训练以 BSD400 和 WED 清晰图作为 target。在完成同步裁剪和几何增强后，等概率
选择：

```text
sigma in {15, 25, 50}
```

然后生成：

```python
noise = torch.randn_like(clean) * (sigma / 255.0)
degraded = clean + noise
```

训练 input 不预先 clamp，使高斯噪声保持正确统计分布；target 保持 `[0, 1]`。
模型输出在训练损失之前也不 clamp。

验证和测试噪声必须可复现。每个样本的随机种子固定由以下字符串的 SHA256 低位
整数派生：

```text
"aio3-v1:<split>:<image_id>:sigma<sigma>"
```

禁止使用 Python 内置 `hash()`，因为它可能跨进程变化。

## 7. 三任务平衡采样

OTS 的退化图数量远大于其他任务，不能简单拼接三个数据集后 shuffle。

每个训练 batch 固定为 12 张，并包含：

```text
4 denoise + 4 derain + 4 dehaze
```

规则如下：

- 三个任务始终按 1:1:1 样本比例优化；
- denoise 内部先均匀抽取 BSD400/WED 的清晰样本，再等概率抽取三个 sigma；
- derain 在 180 个训练 pair 中均匀有放回抽样；
- dehaze 先均匀抽 clear scene，再均匀抽该 scene 的 haze；
- batch 内样本顺序可以随机打乱，但必须保留每个样本的 task 标签；
- 一个训练“epoch”不对应任何原始数据集的完整遍历，训练和调度一律使用
  `global_step`。

若未来模型显存不足，只允许减小 micro-batch 并使用梯度累积，**有效 batch 必须仍为
12，且每次 optimizer step 的三个任务贡献仍为 1:1:1**。改变有效 batch 属于新协议。

### 7.1 当前 Dataset 与平衡采样实现

`aio3_runner.data.AIO3ManifestDataset` 直接读取冻结 manifest。训练模式执行同步 padding、
128裁剪、水平/垂直翻转、90度旋转和在线高斯噪声；验证/测试保持原始分辨率。

`BalancedTaskBatchSampler` 不直接按73859条记录均匀抽样，而是先按任务，再按scene均匀
抽取，最后在该scene内选择退化版本。每个 batch 精确包含4个 denoise、4个 derain、
4个 dehaze。

采样器为每个 `global_step` 生成 `(record_index, sample_seed)` 请求，Dataset 的裁剪、
增强、sigma和噪声全部只由该 seed 决定。因此 worker 数量和预取顺序不影响样本；从
checkpoint 的 `global_step` 重建 sampler 后，下一批请求与不中断训练完全相同。

服务器测试命令：

```bash
cd /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DCNv4
python tests/test_aio3_data.py
```

## 8. 模型公共接口

公共 runner 通过模型 adapter 创建模型。adapter 必须实现：

```python
model = build_model(model_config)
restored_raw = model(degraded)
```

公共约束：

- `degraded` 和 `restored_raw` 的形状相同；
- adapter 负责模型所需的尺寸倍数 padding/crop；
- adapter 不得在前向末尾 clamp；
- runner 不依赖中间特征或模型专属额外输出；
- 所有基线均从随机初始化开始，禁止加载任务预训练或恢复 checkpoint；
- 模型原生的随机初始化方法可以保留，DCNv4 当前零初始化最终残差卷积的行为也保留，
  但初始化方法必须记录到 W&B config；
- 参数量和可训练参数量必须在启动时记录。

DCNv4 的训练前 smoke test 必须再次验证四个 DCNv4 算子的实际输入、输出均为
FP32，并完成 BF16 外层网络的 forward/backward。

## 9. 冻结训练配置

`AIO3-v1` 的主训练配置固定为：

```yaml
protocol: aio3-v1
model: dcnv4_restoration_unet
seed: 3407

data:
  patch_size: 128
  batch_size: 12
  task_samples_per_batch:
    denoise: 4
    derain: 4
    dehaze: 4
  num_workers: 8
  pin_memory: true
  persistent_workers: true

training:
  max_steps: 200000
  precision: bf16
  grad_clip_norm: 1.0
  loss: l1

optimizer:
  name: adamw
  learning_rate: 2.0e-4
  betas: [0.9, 0.999]
  weight_decay: 1.0e-4

scheduler:
  name: warmup_cosine
  warmup_steps: 2000
  min_learning_rate: 1.0e-6

validation:
  interval_steps: 5000
  batch_size: 1

checkpoint:
  interval_steps: 5000
  milestone_interval_steps: 50000

monitoring:
  provider: wandb
  mode: online
  project: aio3-restoration
  group: aio3-v1
  scalar_interval_steps: 50
  media_interval_steps: 10000
  upload_manifest_artifact: true
  upload_best_checkpoint_artifact: true
```

优化损失为所有样本、通道和像素上的平均 L1：

```python
loss = torch.mean(torch.abs(restored_raw - target))
```

本协议不使用 SSIM loss、感知 loss、频率 loss、边缘 loss、任务权重、EMA 或模型专属
loss。以后增加这些内容必须作为单独消融实验，不得覆盖基线结果。

BF16 使用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`，不使用
GradScaler。每次 optimizer step 前将全模型梯度范数裁剪到 1.0。scheduler 按
optimizer step 更新，而不是按数据集 epoch 更新。

warmup cosine 的定义也必须保持一致。以从 0 开始计数的 optimizer update index
`i` 表示下一次参数更新：前 2000 次更新使用 `base_lr * (i + 1) / 2000`，因此第一次
更新使用 `1e-7`，第 2000 次更新达到 `2e-4`；之后使用单周期 cosine，在第 200000 次
更新使用 `1e-6`。optimizer 更新完成后再推进一次 scheduler。checkpoint 中保存的
`completed_steps` 是已经成功完成的 optimizer 更新数，恢复后由 scheduler 状态精确设置
下一次更新的学习率，禁止根据日志或 dataloader batch index 猜测状态。

开发顺序固定为：

1. 100 step smoke test，检查配对、梯度、显存和 checkpoint round-trip；
2. 5000 step pilot，完成一次完整验证并检查预测图；
3. pilot 无异常后从头启动 200000 step 正式训练；
4. 正式比较建议使用种子 `3407/3408/3409`，报告均值与标准差；首轮 DCNv4
   基线允许只运行 3407。

## 10. Checkpoint 与恢复训练

checkpoint 至少保存：

- 模型 `state_dict`；
- DCNv4 架构元数据；
- optimizer 和 scheduler 状态；
- `global_step`、当前最佳验证指标；
- Python、NumPy、PyTorch CPU 和 CUDA RNG 状态；
- 完整冻结配置与协议版本；
- 数据 manifest SHA256；
- runner 和模型仓库 Git commit；
- PyTorch、CUDA、GPU 和 DCNv4 扩展版本信息。
- W&B entity、project、run ID、run name 和本地 run 目录。

文件保留策略：

```text
checkpoints/latest.pth
checkpoints/best_macro_psnr.pth
checkpoints/step_050000.pth
checkpoints/step_100000.pth
checkpoints/step_150000.pth
checkpoints/step_200000.pth
```

普通的 5000-step checkpoint 可以循环保留最近 3 个。恢复训练必须使用
`latest.pth` 并恢复 optimizer、scheduler、global step 和 RNG。只加载模型权重开始
新实验时必须生成新的 run，并明确标记为 initialization，而不是 resume。

每到验证边界，runner 必须先原子保存并回读校验 `latest.pth`，再把
`run_state.json` 写为 `validating/<global_step>`，最后才执行原始分辨率验证。验证失败时，
该 step 的模型、optimizer、scheduler 和 RNG 因而仍可恢复；验证成功后才更新最佳指标、
`best_macro_psnr.pth` 和 `completed/running` 状态。标量窗口落盘时同步刷新
`run_state.json` 的当前 step，不能在长训练期间一直显示 0。

## 11. 验证与最佳模型选择

验证每 5000 optimizer steps 执行一次，使用原始分辨率和 batch size 1，不做随机
增强，不使用 test-time augmentation。

模型预测先转换为 FP32，再仅为指标和可视化执行：

```python
prediction_metric = restored_raw.float().clamp(0.0, 1.0)
```

训练损失仍使用未 clamp 的 `restored_raw`。

验证分别报告：

- denoise PSNR/SSIM：sigma 15、25、50；
- denoise 平均：三个 sigma 指标的算术平均；
- derain PSNR/SSIM；
- dehaze PSNR/SSIM；
- task macro PSNR/SSIM。

任务宏平均定义为：

```text
denoise_psnr = mean(psnr_sigma15, psnr_sigma25, psnr_sigma50)
macro_psnr = mean(denoise_psnr, derain_psnr, dehaze_psnr)
```

SSIM 使用相同方式计算 macro。`best_macro_psnr.pth` 只按验证集 macro PSNR 更新。
正式测试集不得用于选择 checkpoint 或调整超参数。

## 12. 正式测试协议

训练完成后只对 `best_macro_psnr.pth` 执行正式测试：

| 任务 | 测试集 | 条件 |
|---|---|---|
| denoise | BSD68 | 固定噪声 sigma 15、25、50，共 68 x 3 个条件 |
| derain | Rain100L | 审计通过的 100 对 |
| dehaze | SOTS Outdoor | 500个严格配对样本，492个唯一target |

测试要求：

- 使用原始分辨率；
- batch size 为 1；
- 不做 resize、随机增强或 test-time augmentation；
- 优先使用整图推理；A800 80GB 无法整图推理时才允许 tiled inference；
- 一旦使用 tile，所有参与 `AIO3-v1` 比较的模型必须使用相同 tile size、overlap
  和融合方法，并在结果中记录；
- 输出文件名必须保留样本唯一 ID，不能按 dataloader 序号命名；
- 不覆盖已有预测目录。

正式训练完成后，在同一干净代码 commit 上执行：

```bash
python -m aio3_runner.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/best_macro_psnr.pth" \
  --num-workers 4
```

评测入口只接受 `run_kind=formal` 且状态为 `completed` 的
`best_macro_psnr.pth`；smoke/pilot checkpoint 会被拒绝，防止提前使用测试集。测试 gallery
在推理前仅依据 sample ID 哈希固定选择：每个噪声强度 2 张、去雨 4 张、去雾 4 张，
总计 14 张，不得依据模型结果重新挑图。804 张预测全部保存在本地，但 W&B 只上传数值表
和这 14 组 gallery。

## 13. 指标定义

主指标为 RGB PSNR 和 RGB SSIM：

- 计算范围 `[0, 1]`，`data_range=1.0`；
- prediction 在指标前 clamp，target 不做额外量化；
- 使用 FP32 张量计算，不从保存后的 8-bit PNG 重新计算；
- 不转 Y 通道，不裁边；
- PSNR 先逐图计算，再对数据集求算术平均；
- SSIM 使用 11 x 11 Gaussian window、sigma 1.5，三个 RGB 通道平均；
- SSIM 常数固定为 `K1=0.01`、`K2=0.03`，使用总体协方差（不做无偏校正）；
- SSIM 通过分组 `conv2d` 逐通道计算，窗口归一化为和 1，使用
  `padding=0` 的 valid 区域，最后对通道和有效空间位置求平均；
- 同时保存样本级指标，便于复核平均值。

不得把不同任务的所有像素合并后计算一个全局 PSNR。最终报告必须列出每个数据集和
每个噪声强度，macro 只能作为辅助总览，不能替代分任务结果。

## 14. W&B 训练监控与可视化规范

### 14.1 定位与基本原则

W&B 用于远程实时监控、跨 run 对比和定性可视化，但不是唯一的实验记录。训练代码必须
同时写入本地 JSONL/CSV；W&B 暂时不可用时，不得丢失训练指标，也不得改变模型梯度、
采样顺序或 checkpoint 行为。

不默认调用 `wandb.watch(model)`。全模型权重和梯度 histogram 会给 DCNv4 训练增加
额外 hook、显存、CPU 和网络开销。本协议手动记录总梯度范数，并只在验证阶段记录恢复
残差 histogram。

`AIO3-v1` 不使用 W&B Sweep 自动搜索超参数。本文训练参数已经冻结；任何 sweep 结果
只能作为下一版协议或单独消融实验。

### 14.2 安装、登录与本地目录

W&B SDK 必须安装在 `general-decomp` 训练环境中，并将实际版本写入环境规格文件、
`environment.json` 和 W&B config。服务器使用 Python 3.9，因此 `environment.yml` 固定
`wandb==0.25.1`；当前 W&B 0.28.x 已要求 Python 3.10 或更高，不能直接用于该环境。
SDK 版本变化通常不改变训练数学协议，但必须记录，便于排查日志行为变化。版本兼容信息
以 [W&B 0.25.1 PyPI 元数据](https://pypi.org/project/wandb/0.25.1/) 为准。

API key 只通过以下方式之一提供：

```bash
wandb login
# 或在任务调度器/安全环境中设置 WANDB_API_KEY
```

安装与连通性检查命令固定为：

```bash
python -m pip install 'wandb==0.25.1'
python -c "import wandb; print(wandb.__version__)"

python -m aio3_runner.wandb_check \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1
```

若账号默认 entity 不正确，显式增加 `--entity <account-or-team>`。只有 online 连通性测试
成功且网页可见后，才启动 online smoke run；网络不可用时使用 `--mode offline` 验证本地
写入，并在正式命令中固定 `--wandb-mode offline`。

禁止把真实 API key 写入源码、shell 脚本、YAML、checkpoint、日志或 Git。运行前由 runner
创建 `${RUN_DIR}/wandb`，并把 W&B run、cache 和 staging 路径设置到输出盘；以下环境变量
形式可用于人工复核或覆盖调度环境：

```bash
export WANDB_PROJECT=aio3-restoration
export WANDB_MODE=online
export WANDB_DIR="${RUN_DIR}"
export WANDB_CACHE_DIR="${OUTPUT_ROOT}/.wandb_cache"
export WANDB_DATA_DIR="${OUTPUT_ROOT}/.wandb_staging"
```

`WANDB_ENTITY` 由用户账号或团队决定，不写死在公共配置里；启动时必须解析并保存实际
entity。`WANDB_DIR`、cache 和 staging 均位于输出盘，不能落到源码仓库、原始数据目录
或空间较小的系统盘。

在 100-step smoke test 前先运行一个只记录单个标量的 W&B 连通性测试，并在网页确认
run 可见。正式训练使用 `WANDB_MODE=online` 才能远程实时监控。

100-step DCNv4 smoke run 的统一入口为：

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --run-kind smoke \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online
```

如需指定账号或团队，增加 `--wandb-entity <account-or-team>`。恢复命令只接受同一运行的
`checkpoints/latest.pth`，W&B mode、entity、run ID 和全部训练配置均从 checkpoint 与
`config.yaml` 读取：

```bash
python -m aio3_runner.train --resume "${RUN_DIR}/checkpoints/latest.pth"
```

在进入 5000-step pilot 前，另建一个 100-step smoke run 做一次真实恢复验收。首次启动增加
`--pause-at-step 50`；runner 会在第 50 次 optimizer 更新及标量落盘完成后，原子保存
`latest.pth`，把 `run_state.json` 标记为 `paused`，随后正常关闭同一个 W&B run：

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --run-kind smoke \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  --pause-at-step 50
```

记录控制台输出的 `RUN_DIR`，确认状态和 checkpoint 都停在 50，再执行上述 `--resume`
命令跑至 100。`--pause-at-step` 是执行控制而非实验超参数，不写入冻结 config；暂停点
必须小于 `max_steps`，并与标量记录间隔对齐，从而避免丢弃半个聚合窗口。恢复后必须确认：

1. `run_state.json` 从 `paused/50` 变为 `completed/100`；
2. `latest.pth` 的 `global_step` 与 scheduler `completed_steps` 均为 100；
3. `wandb_run_id.txt`、checkpoint 和恢复前后的 W&B URL 使用同一个 run ID；
4. W&B 历史的训练步为连续的 10、20、...、100，且只存在一个 run；
5. 第 100 step 的完整验证、14 行固定样本 Table 和最佳模型 Artifact 均存在。

### 14.3 Run 组织方式

W&B 固定组织方式：

```text
project: aio3-restoration
group: aio3-v1
job_type: train
run_name: dcnv4-unet-seed<seed>-<YYYYMMDD-HHMMSS>
tags: [aio3-v1, dcnv4, baseline, signed-residual]
```

一个随机种子的一次正式训练对应一个 W&B run。100-step smoke test 和 5000-step pilot
必须使用不同 run，并增加 `smoke` 或 `pilot` tag，不能续接到正式 run。

首次启动正式训练时生成唯一 `wandb_run_id`，立即写入：

```text
${RUN_DIR}/wandb_run_id.txt
${RUN_DIR}/run_state.json
```

同一个 ID 也必须保存在每个 checkpoint 中。首次启动使用该 ID 创建新 run；从
`latest.pth` 恢复时读取 checkpoint 中的 ID，并使用同一 ID 严格续跑。不得仅依赖当前
工作目录自动寻找旧 run，也不得在恢复训练时创建第二个 W&B run。

参考初始化逻辑：

```python
run = wandb.init(
    entity=resolved_entity,
    project="aio3-restoration",
    group="aio3-v1",
    job_type="train",
    name=run_name,
    id=wandb_run_id,
    resume="must" if resume_training else "never",
    dir=str(run_dir),
    config=resolved_config,
    tags=["aio3-v1", "dcnv4", "baseline", "signed-residual"],
)
```

只有主进程或 rank 0 可以初始化和写入 run。不能让多个进程使用同一个 run ID 并发
写入。单卡 A800 训练也沿用这一规则，便于以后扩展分布式训练。

### 14.4 统一 global step 横轴

W&B 内部 step 会随每次 `run.log()` 调用增加，训练标量、验证标量和图像的日志调用
次数不同，因此所有训练曲线必须显式使用 `global_step` 作为横轴：

```python
run.define_metric("global_step")
run.define_metric("train/*", step_metric="global_step")
run.define_metric("diagnostics/*", step_metric="global_step")
run.define_metric("system/*", step_metric="global_step")
run.define_metric("val/*", step_metric="global_step")
run.define_metric("val/macro/psnr", step_metric="global_step", summary="max")
run.define_metric("val/macro/ssim", step_metric="global_step", summary="max")
```

`wandb==0.25.1` 的单星号 namespace 定义不能可靠覆盖更深层的指标路径。因此 runner
还必须逐一对 `val/denoise/sigma*/{psnr,ssim,images}`、各任务验证指标、分任务残差诊断
以及 `test/*` 下的多层指标调用精确的 `define_metric(...,
step_metric="global_step")`。W&B 图表横轴显示为 `Step` 说明绑定失败，正式训练前必须修复；
正确横轴应显示为 `global_step`。

每一次 `run.log()` 都必须在同一个字典中包含当前 `global_step`。不得同时依赖 W&B
内部 step 或把 dataloader batch index 当作训练 step。最佳指标还要显式写入：

```text
run.summary["best/val_macro_psnr"]
run.summary["best/val_macro_ssim"]
run.summary["best/global_step"]
```

### 14.5 启动配置与可复现信息

启动时上传到 W&B config：

- 完整解析后的训练配置，不仅是命令行显式参数；
- 协议版本和本协议文档的 SHA256；
- 模型参数量、可训练参数量和初始化方法；
- 每个 split 的样本/场景计数和全部 manifest SHA256；
- 模型、runner Git commit 和工作树是否干净；
- Python、W&B、PyTorch、torchvision、CUDA、Pillow、GPU 和 DCNv4 扩展版本；
- BF16 autocast 与 DCNv4 FP32 隔离状态；
- 当前是否使用 tiled inference；
- SOTS 是否为500个严格配对样本及其唯一target数量；
- hostname、随机种子、有效 batch、worker 数量和启动命令。

config 在 run 创建后视为只读。若恢复训练时 checkpoint config、manifest SHA256、模型
commit 或协议版本不一致，runner 必须拒绝恢复，不能用 W&B config 覆盖本地事实。

### 14.6 训练标量

训练标量每 50 个 optimizer steps 记录一次。记录值为最近 50 steps 的样本加权平均，
不能只记录最后一个 batch。一次性调用 `run.log()` 写入同一 global step：

```text
global_step
train/loss
train/denoise_l1
train/derain_l1
train/dehaze_l1
train/learning_rate
train/grad_norm
train/step_time_seconds
train/images_per_second
train/samples_denoise
train/samples_derain
train/samples_dehaze
system/gpu_memory_allocated_gib
system/gpu_memory_reserved_gib
```

三个 `train/samples_*` 在每个 optimizer step 都应等价于 4；日志窗口内应各为 200。
如果不相等，说明平衡 batch sampler 失效，正式训练必须停止。

W&B 会自动采集部分 CPU/GPU 利用率；PyTorch 显存和吞吐量仍由 runner 手动记录，作为
与实际训练 step 对齐的性能指标。显示在网页上的平滑曲线只用于观察，论文或表格中的
数值必须来自未平滑的本地指标。

### 14.7 有符号残差诊断

为持续检查 DCNv4 残差头没有重新退化为非负输出，对
`signed_residual = restored_raw - degraded` 记录：

```text
diagnostics/residual_mean
diagnostics/residual_std
diagnostics/residual_min
diagnostics/residual_max
diagnostics/residual_negative_fraction
diagnostics/residual_positive_fraction
diagnostics/residual_near_zero_fraction
diagnostics/prediction_below_zero_fraction
diagnostics/prediction_above_one_fraction
```

其中 near-zero 固定定义为 `abs(residual) <= 1e-6`。正负比例是诊断信号，不是优化
目标，禁止为追求某个比例额外增加 loss。不同任务需要的残差分布本来就可能不同，验证
阶段还要分别记录：

```text
diagnostics/denoise/residual_negative_fraction
diagnostics/derain/residual_negative_fraction
diagnostics/dehaze/residual_negative_fraction
```

如果非零残差中负值长期精确为 0，应立即检查最终激活、clamp 和 adapter，而不是继续
训练。验证媒体步骤额外记录一个固定样本集合的 `wandb.Histogram`，用于观察残差是否
塌缩、偏置或出现异常长尾。

### 14.8 验证指标

每 5000 steps 验证一次，并在一次 `run.log()` 中记录：

```text
val/denoise/sigma15/psnr
val/denoise/sigma15/ssim
val/denoise/sigma25/psnr
val/denoise/sigma25/ssim
val/denoise/sigma50/psnr
val/denoise/sigma50/ssim
val/denoise/mean/psnr
val/denoise/mean/ssim
val/derain/psnr
val/derain/ssim
val/dehaze/psnr
val/dehaze/ssim
val/macro/psnr
val/macro/ssim
```

同时更新 `run.summary` 中的最佳值和最佳 step。W&B 显示值必须来自与本地
`validation/metrics_step_<step>.json` 相同的聚合结果，禁止在 W&B 日志层进行另一套
平均。

### 14.9 固定样本可视化

可视化样本 ID 在 manifest 创建时冻结，不能挑选当前模型表现较好的图片：

- denoise：每个 sigma 固定 2 张，共 6 张；
- derain：固定 4 张；
- dehaze：固定 4 张。

每 10000 steps 记录一次，共 14 个样本。每个样本必须展示：

1. degraded input；
2. `prediction.clamp(0, 1)`；
3. clean target；
4. absolute error；
5. signed residual；
6. 当前 PSNR、SSIM、残差均值和负残差比例。

显示规则必须固定，避免不同 step 的颜色尺度产生误导：

- input/prediction/target 仅为显示 clamp 到 `[0, 1]`；
- absolute error 固定显示范围 `[0, 0.25]`，超过部分饱和；
- signed residual 固定显示范围 `[-0.25, 0.25]`，0 映射到中性色；
- 训练和指标张量不能被可视化转换原地修改；
- 同一 sample ID 在所有 step 使用相同 W&B key 和相同排列顺序。

使用一个 `wandb.Table` 组织固定样本，列固定为：

```text
global_step, task, sigma, sample_id,
input, prediction, target, absolute_error, signed_residual,
psnr, ssim, residual_mean, residual_negative_fraction
```

`wandb.Image` 只接收 CPU 上已经转换好的显示图，不直接持有 GPU tensor。Table 每个媒体
step 只包含上述 14 行，避免把完整验证集图像重复上传。

### 14.10 W&B Workspace 面板布局

项目 Workspace 固定整理为以下六组，所有 run 使用同一布局：

1. **Training health**：总 L1、三任务 L1、学习率、梯度范数；
2. **Task balance**：三任务样本数，及时发现 sampler 比例错误；
3. **Validation**：三档去噪、去雨、去雾及 macro PSNR/SSIM；
4. **Signed residual diagnostics**：残差均值/标准差/正负比例、越界预测比例、histogram；
5. **Qualitative restoration**：固定样本 Table 和图像演化；
6. **Performance**：显存、step time、吞吐量和 W&B 自动采集的 GPU utilization。

跨 run 对比时按 `group=aio3-v1` 过滤，以 seed、model commit 和 manifest SHA256 作为
Runs Table 列。禁止用网页平滑后的极值选择 checkpoint，checkpoint 选择只由 runner 的
未平滑验证 macro PSNR 决定。

### 14.11 Artifact 策略

W&B Artifact 只保存可复现所需的小型数据描述和最佳模型，不上传 AIO-3 原始图像或
全部预测 PNG。

训练开始时创建 dataset artifact：

```text
name: aio3-v1-manifests
type: dataset
contents:
  protocol Markdown
  data_audit.json
  train/val/test manifests
  resolved config
```

Artifact metadata 必须包含 manifest SHA256 和数据根目录标识。它描述本地数据，但不
复制 BSD/WED/Rain/OTS/SOTS 原始图像。

训练结束时创建 model artifact：

```text
name: aio3-v1-dcnv4-unet-seed<seed>
type: model
aliases: [best, seed<seed>]
contents:
  best_macro_psnr.pth
  config.yaml
  environment.json
  data_audit.json
  final validation metrics
```

正式测试结束后创建 evaluation artifact，只包含 `metrics.json`、`metrics.csv`、
`per_image_metrics.csv` 和一个不超过 14 张图的固定测试 gallery。完整预测图仍保存在
`${RUN_DIR}/test/predictions`。Artifact 上传不得替代本地 checkpoint；网络中断时先保证
本地文件完整，之后再同步。

### 14.12 断网、离线与异常处理

服务器访问 GitHub 曾出现超时，因此必须预期 W&B 网络也可能不稳定：

1. 在线模式初始化失败时，在任何 optimizer step 开始前退出并给出明确错误；
2. 用户确认后才用 `--wandb-mode offline` 创建新的 run 目录；失败的 online 初始化目录
   不得改写 config 后复用；
3. offline run 仍写入 `${RUN_DIR}/wandb`，并保存相同的 run ID；
4. 网络恢复后同步具体 run 目录，而不是扫描并误传其他实验；
5. 在线训练中单次日志通信异常不能中断梯度更新，本地 JSONL 继续写入并在控制台告警；
6. 训练结束时在 `finally` 中调用 `run.finish()`，同时保证 checkpoint 先落盘。

当前 W&B SDK 在上传发生不可恢复错误后仍允许本地继续记录；runner 额外将每次 W&B API
异常写入 `logs/wandb_errors.jsonl`，所有 W&B 调用前后恢复 Python、NumPy、PyTorch CPU
和 CUDA RNG 状态。W&B 的 online/offline 与 resume 语义以官方
[init API](https://docs.wandb.ai/models/ref/python/functions/init) 和
[resume 指南](https://docs.wandb.ai/models/runs/resuming) 为准。

同步命令形式为：

```bash
wandb sync "${RUN_DIR}/wandb/<offline-run-directory>"
```

不得自动执行 `wandb sync --clean`，因为它会清理本地数据。确认网页数据和 Artifact
完整之前，保留整个 W&B 本地目录。

### 14.13 正式测试的 W&B 记录

正式测试作为同一训练 run 的最终阶段记录，并增加 `test/*` 指标；如果测试在另一台机器
或独立时间执行，则创建 `job_type=evaluation` 的新 run，并使用训练 model artifact 作为
输入，不能把测试曲线伪装成训练续跑。

测试结束记录：

```text
test/bsd68/sigma15/psnr
test/bsd68/sigma15/ssim
test/bsd68/sigma25/psnr
test/bsd68/sigma25/ssim
test/bsd68/sigma50/psnr
test/bsd68/sigma50/ssim
test/bsd68/mean/psnr
test/bsd68/mean/ssim
test/rain100l/psnr
test/rain100l/ssim
test/sots_outdoor/psnr
test/sots_outdoor/ssim
test/macro/psnr
test/macro/ssim
test/total_runtime_seconds
```

完整逐图指标以一个纯数值 `wandb.Table` 上传，包含 dataset、task、sigma、sample ID、
PSNR、SSIM 和推理时间。不要在逐图表中附带全部高分辨率图片；图像只使用固定 gallery。
同一服务器、同一 run 目录的正式测试恢复训练 run ID 并追加 `test/*`；evaluation Artifact
包含 `metrics.json`、两类 CSV、gallery 描述和固定 gallery，不得包含完整 predictions。

### 14.14 官方接口依据

本方案使用的 run 初始化、显式 run ID 续跑、自定义 global-step 横轴、Image/Table、
Artifact、环境变量和 offline sync 均来自 W&B 官方文档：

- [Initialize a run](https://docs.wandb.ai/models/ref/python/functions/init)
- [Resume a run](https://docs.wandb.ai/models/runs/resuming)
- [Customize log axes](https://docs.wandb.ai/models/track/log/customize-logging-axes)
- [Log tables](https://docs.wandb.ai/models/track/log/log-tables)
- [Artifacts overview](https://docs.wandb.ai/models/artifacts)
- [Environment variables](https://docs.wandb.ai/models/track/environment-variables)
- [wandb sync](https://docs.wandb.ai/models/ref/cli/wandb-sync)

## 15. 输出目录与结果文件

单次运行目录固定为：

```text
${OUTPUT_ROOT}/dcnv4_unet/<run_name>/
├── config.yaml
├── environment.json
├── AIO3_TRAINING_EVALUATION_PROTOCOL.md
├── wandb_run_id.txt
├── wandb_state.json
├── run_state.json
├── train_metrics.jsonl
├── validation_metrics.jsonl
├── manifests/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── visual_samples.json
│   └── data_audit.json
├── checkpoints/
├── logs/
├── wandb/
├── validation/
└── test/
    ├── state.json
    ├── metrics.json
    ├── metrics.csv
    ├── per_image_metrics.csv
    ├── gallery_selection.json
    ├── gallery.json
    ├── gallery/
    └── predictions/
        ├── BSD68_sigma15/
        ├── BSD68_sigma25/
        ├── BSD68_sigma50/
        ├── Rain100L/
        └── SOTS-outdoor/
```

`metrics.json` 必须包含协议版本、checkpoint SHA256、manifest SHA256、Git commits、
数据集实际样本数、每项均值和运行时间。`metrics.csv` 用于模型间汇总，
`per_image_metrics.csv` 用于逐样本复查。

## 16. 训练启动前强制检查

以下任一项失败时不得启动正式训练：

1. manifest 审计通过，RainTrainL=200 对、Rain100L=100 对；
2. train/val/test 的 scene ID 交集为空；
3. 任取样本的 input/target 尺寸一致且均为 3 通道；
4. 一个 batch 恰好包含 4/4/4 个三任务样本；
5. 固定种子下验证和测试噪声逐像素可复现；
6. DCNv4 127 x 191 任意尺寸 BF16 forward/backward 测试通过；
7. 四个 DCNv4 算子的实际输入输出均为 FP32；
8. 100-step smoke test 无 NaN/Inf，所有应训练参数具有有限梯度；
9. checkpoint 保存再加载后，固定输入的输出一致；
10. W&B 连通性 smoke run 能记录配置、global-step 标量、固定验证 Table 和图片；
11. 暂停并恢复 smoke run 后沿用同一个 W&B run ID，历史曲线不产生第二个 run；
12. 有符号残差诊断同时能够观察到正值与负值，且可视化采用固定色阶；
13. 输出目录不在 Git 仓库和原始数据目录内；
14. `config.yaml`、data manifest 和代码 commit 已冻结并被记录。
15. 5000-step pilot 的训练、验证、checkpoint、W&B 和 70 张固定媒体审计通过；
16. 人工复查 14 组固定验证样本，重点检查 sigma50 残余噪声、去雨内容误删和去雾
    亮度/颜色偏差；
17. 正式测试入口的合成测试通过，并确认它拒绝非 formal checkpoint。

## 17. 协议变更规则

以下变化会破坏 `AIO3-v1` 可比性，必须升级协议版本：

- 使用 `WED/noisy` 或改变噪声类型/强度/随机种子规则；
- 改变 train/val/test 划分；
- 改变三任务采样比例、有效 batch 或 patch size；
- 改变 loss、optimizer、学习率或总 optimizer steps；
- 在某一模型上单独启用 EMA、预训练权重、TTA 或不同 tile 策略；
- 改变 PSNR/SSIM 通道、裁边、clamp 或实现；
- 改变 SOTS 的500-pair测试manifest或其input-target映射。

修复不改变数学行为的工程错误可以保留协议版本，但必须记录修复 commit，并重新运行
受影响的实验。任何影响数据、梯度或指标数值的修复都应从头训练并生成新的 run。
