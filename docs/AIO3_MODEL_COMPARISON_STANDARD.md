# AIO-3 同功能网络统一训练、评测与 W&B 规范

## 1. 文档目的与效力

本文档用于指导不同图像恢复网络在完全一致的 AIO-3 条件下完成去噪、去雨和去雾联合
训练，并生成可直接比较的验证、测试和 W&B 记录。适用对象包括 U-Net、Uformer、
Restormer 以及以后加入退化感知模块的模型。

协议标识固定为 `AIO3-v1`。除模型 adapter、模型结构、模型自身初始化及模型无法在
BF16 下运行时的局部精度隔离外，本文规定的数据、采样、增强、loss、optimizer、
scheduler、训练步数、验证、测试、指标和 checkpoint 选择方法均为强制项。

本文是其他模型接入时的执行规范。更完整的设计依据和边界条件见
[`AIO3_TRAINING_EVALUATION_PROTOCOL.md`](AIO3_TRAINING_EVALUATION_PROTOCOL.md)。两者发生
歧义时，以完整协议、冻结 manifest 及公共 runner 实现为准。

规范状态：

```text
protocol: AIO3-v1
document revision: 2026-08-10
reference baseline training commit: 6a2c14f92770506fe3b2558ec4072037189b1ea9
reference repository: Chivioletlly/DCNv4fortest
reference branch: aio3-dcnv4-signed-residual
```

## 2. 公平比较的核心原则

所有标记为 `AIO3-v1` 的正式结果必须同时满足：

1. 使用同一套冻结 manifest 及完全相同的 SHA256；
2. 三任务按 `1:1:1` 采样，每个 optimizer step 的有效 batch 为12；
3. 使用相同的128训练 patch及同步几何增强；
4. 使用同一套在线高斯噪声规则和固定验证/测试噪声；
5. 从随机初始化开始，不加载任务预训练权重；
6. 统一使用像素 L1、AdamW、warmup cosine及200000个 optimizer steps；
7. 不使用 EMA、TTA、感知损失、SSIM损失、任务权重或模型专属训练技巧；
8. 验证和测试统一使用原始分辨率、batch size 1及公共指标代码；
9. 只按验证集 task-macro PSNR选择 checkpoint；
10. 测试集只在正式训练完成后评测，不得用于模型选择或调参；
11. W&B、JSONL、checkpoint和Git状态足以复现实验；
12. 所有协议差异必须改用新的协议名称，不能继续声称为 `AIO3-v1`。

参数量、FLOPs、推理速度和显存不是必须相同的公平条件，但必须报告。模型容量不同本身
就是比较结果的一部分。

## 3. 固定服务器目录

```text
PROJECT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation
DATA_ROOT=${PROJECT_ROOT}/data/AIO3
MODEL_ROOT=${PROJECT_ROOT}/all-in-one-model
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/AIO3/aio3-v1
MANIFEST_DIR=${OUTPUT_ROOT}/manifests
```

目录布局：

```text
Unet4Degradation/
├── data/AIO3/                         # 原始数据，只读
├── all-in-one-model/
│   ├── DCNv4/                         # 公共 runner 和参考模型
│   ├── Uformer/
│   └── <model_id>/                    # 其他网络代码
└── outputs/AIO3/aio3-v1/
    ├── manifests/                     # 所有模型共享的冻结数据描述
    ├── dcnv4_unet/
    ├── uformer/
    └── <model_id>/
```

禁止将 manifest、checkpoint、日志、W&B本地文件或预测图写入原始数据目录和源码仓库。
每个模型必须拥有独立的 `<model_id>` 输出目录。

## 4. 冻结数据与 Manifest

### 4.1 原始数据用途

| 数据集 | 用途 | AIO3-v1处理方式 |
|---|---|---|
| BSD400 | 去噪训练 | 只使用清晰图，在线合成噪声 |
| WED/gt | 去噪训练和验证 | 只使用gt，在线或固定种子合成噪声 |
| WED/noisy | 不使用 | 噪声元数据不明确 |
| BSD68 | 去噪正式测试 | 每图固定生成sigma 15/25/50 |
| RainTrainL | 去雨训练和验证 | `rain-N`严格匹配`norain-N` |
| Rain100L | 去雨正式测试 | `rain/rain-N`严格匹配`gt/norain-N` |
| OTS | 去雾训练和验证 | 先均匀抽clear scene，再抽haze版本 |
| SOTS Outdoor | 去雾正式测试 | 500组严格input-target pair |

必须排除：

- `WED/noisy`；
- `RainTrainL/rainregion-*`；
- `OTS/depth`；
- `OTS/haze/part1.zip`至`part4.zip`；
- `.ipynb_checkpoints`、隐藏临时文件及非图像文件。

### 4.2 当前冻结文件

所有模型必须直接复用以下文件，不得各自重新划分：

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `train.jsonl` | 73859 | `bd153a3b211957184de7b6171d6bc06a48f321b1c571906604d869b1aa19ca7e` |
| `val.jsonl` | 420 | `9c66c4c74a0279858ecab33df998b8eb55d6df021d2e59bbd1c253830ab3f50b` |
| `test.jsonl` | 804 | `7d80fd0af7aeaac2b6e901e20e71a744d7d705f98641f54913aa278e12c2b63a` |
| `data_audit.json` | - | `2959e402ecdb76172b9fe9bba3fae13c090348379dcd33898992abdc198e06b8` |
| `visual_samples.json` | - | `62e9f6e761e3db2c23895958f3707414a59baac30840919044fe6bf848ff628b` |

划分统计：

| split | denoise | derain | dehaze | 总计 |
|---|---:|---:|---:|---:|
| train | 5044 | 180 | 68635 | 73859 |
| val | 300 | 20 | 100 | 420 |
| test | 204 | 100 | 500 | 804 |

训练 manifest 中行数不代表采样权重。OTS虽然拥有大量haze文件，公共 sampler仍先均匀
选择任务和scene，防止去雾任务支配训练。

### 4.3 数据审计

manifest已经生成时，只做哈希验证，不要加`--overwrite`。首次部署到一台新服务器时可
用参考 runner重新执行完整审计：

```bash
cd ${MODEL_ROOT}/DCNv4

python -m aio3_runner.prepare_data \
  --data-root "${DATA_ROOT}" \
  --output-dir "${MANIFEST_DIR}"
```

正式审计禁止使用`--skip-image-verification`。任何哈希变化都意味着数据协议发生变化，
不得与本文参考结果直接比较。

## 5. 数据加载、增强与噪声

### 5.1 公共图像处理

- Pillow解码并显式转换为RGB；
- 张量布局为`[C,H,W]`，图像基础范围为`[0,1]`；
- 完整训练图像不resize；
- input和target执行完全同步的空间变换；
- 训练patch固定为`128 x 128`；
- 小图先reflection padding，无法reflection时才replicate padding；
- 水平翻转概率0.5；
- 垂直翻转概率0.5；
- 等概率选择`0/90/180/270`度旋转；
- 禁止颜色抖动、随机锐化及只作用于一侧的几何增强。

### 5.2 在线去噪

去噪训练在同步裁剪和增强后等概率选择：

```text
sigma in {15, 25, 50}
```

并执行：

```python
noise = torch.randn_like(clean) * (sigma / 255.0)
degraded = clean + noise
```

`degraded`不得预先clamp，因此它允许少量超出`[0,1]`。target保持`[0,1]`。验证和测试
噪声必须由manifest中的固定seed复现，禁止使用Python内置`hash()`。

### 5.3 平衡采样与恢复可复现性

每个有效训练batch固定为：

```text
batch size 12 = 4 denoise + 4 derain + 4 dehaze
```

具体规则：

- denoise均匀抽清晰scene，sigma再等概率抽取；
- derain在180个训练pair中均匀有放回抽样；
- dehaze先均匀抽clear scene，再均匀抽该scene的haze版本；
- crop、增强、sigma和噪声全部由`global_step`派生的sample seed决定；
- DataLoader worker数和预取顺序不得改变样本；
- 恢复训练后下一批样本必须与未中断训练完全一致。

如果模型显存不足，可以采用micro-batch加梯度累积，但每次optimizer update的有效
batch仍必须为12，且三个任务各贡献4张。必须在合规报告中记录micro-batch和累积次数。

## 6. 模型接入契约

其他网络只能替换模型adapter和模型专属checkpoint元数据校验，不得复制后悄悄修改公共
数据、loss、指标、训练循环或评测实现。

adapter至少提供：

```python
model = build_model(model_config)
restored_raw = model(degraded)
```

强制接口：

- 输入和输出均为`[B,3,H,W]`且空间尺寸完全相同；
- adapter负责网络所需尺寸倍数的padding，并裁回原始尺寸；
- `restored_raw`是最终恢复图，不得在模型末尾或loss前clamp；
- 公共runner不依赖模型中间特征或额外监督头；
- 参数保持FP32，训练前记录总参数量和可训练参数量；
- 所有模型从随机初始化开始，不加载ImageNet或恢复任务预训练；
- 模型原生初始化可以保留，但必须写入config和W&B；
- 模型必须支持任意验证/测试分辨率，或由adapter无损padding/crop；
- checkpoint必须保存足以严格重建架构的metadata；
- 不允许以测试集表现为依据修改adapter。

默认使用BF16 autocast。某些算子不支持BF16时，允许只对该算子关闭autocast并以FP32
运行，但必须记录算子名称、精度边界和验证测试。禁止为了某个模型静默改成全FP32后仍
不加说明地与BF16结果比较。

## 7. 冻结训练配置

```yaml
protocol: aio3-v1
seed: 3407

data:
  patch_size: 128
  effective_batch_size: 12
  task_samples_per_optimizer_step:
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
  loss: mean_pixel_l1

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
  native_resolution: true

checkpoint:
  interval_steps: 5000
  milestone_interval_steps: 50000

monitoring:
  provider: wandb
  project: aio3-restoration
  group: aio3-v1
  scalar_interval_steps: 50
  media_interval_steps: 10000
```

损失只有：

```python
loss = torch.mean(torch.abs(restored_raw - target))
```

禁止在`AIO3-v1`基线中加入：

- SSIM、感知、频率、边缘或对抗损失；
- 三任务loss权重；
- EMA；
- 预训练权重；
- TTA；
- 根据模型单独改变patch、有效batch、训练步数或学习率；
- 训练时对prediction执行clamp。

BF16使用`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`，不使用
GradScaler。`clip_grad_norm_`阈值为1.0；W&B中的`train/grad_norm`记录裁剪前范数，因而
允许大于1。

### 7.1 学习率精确定义

前2000次optimizer update线性warmup：

```text
lr(i) = 2e-4 * (i + 1) / 2000,  i = 0...1999
```

第一次更新为`1e-7`，第2000次达到`2e-4`。之后单周期cosine下降，并在第200000次
更新使用`1e-6`。scheduler按optimizer update推进，禁止按epoch推进。

## 8. 固定开发与正式训练顺序

| run kind | steps | warmup | 标量 | 验证 | 媒体 | 目的 |
|---|---:|---:|---:|---:|---:|---|
| smoke | 100 | 100 | 10 | 100 | 100 | 配对、梯度、显存、checkpoint、W&B |
| pilot | 5000 | 2000 | 50 | 5000 | 5000 | 收敛趋势和14组视觉检查 |
| formal | 200000 | 2000 | 50 | 5000 | 10000 | 可报告正式结果 |

执行顺序：

1. 数据和模型单元测试；
2. 100-step smoke；
3. 独立的暂停/恢复smoke验收；
4. 从头启动5000-step pilot；
5. 人工检查pilot的14组固定样本；
6. 从头启动200000-step formal，禁止从pilot权重继续；
7. formal完成后选择最佳验证checkpoint；
8. 最后且仅最后运行一次冻结测试集。

首轮模型比较允许只运行seed 3407。用于论文主表时建议运行`3407/3408/3409`，报告均值
和标准差。不同seed必须使用独立run目录和W&B run。

## 9. Checkpoint与恢复

checkpoint至少保存：

- model、optimizer和scheduler state；
- global step及最佳验证指标；
- Python、NumPy、PyTorch CPU和CUDA RNG；
- 模型构建metadata和参数量；
- 完整config、协议版本和manifest SHA256；
- 模型仓库Git commit及dirty状态；
- Python、PyTorch、CUDA、GPU和模型扩展版本；
- W&B entity、project、run ID及run目录。

固定文件：

```text
checkpoints/latest.pth
checkpoints/best_macro_psnr.pth
checkpoints/step_050000.pth
checkpoints/step_100000.pth
checkpoints/step_150000.pth
checkpoints/step_200000.pth
```

每5000 step必须先原子保存并回读`latest.pth`，再开始完整验证。最佳模型只按未平滑的
`val/macro/psnr`更新。正式测试只接受`best_macro_psnr.pth`。

恢复必须使用同一run的`latest.pth`并恢复全部状态：

```bash
python -m aio3_runner.train \
  --resume "${RUN_DIR}/checkpoints/latest.pth"
```

恢复时模型commit、config或manifest哈希不一致必须拒绝启动。W&B必须沿用原run ID，不能
生成第二条曲线。

## 10. 验证与指标

验证集为420个原始分辨率条件：

- denoise：100张WED清晰图 × sigma 15/25/50，共300；
- derain：RainTrainL固定20对；
- dehaze：OTS固定100 scene，每scene一个固定haze版本。

指标必须直接复用公共`aio3_runner.metrics`，不得换成skimage、torchmetrics或模型仓库
自带的另一种PSNR/SSIM实现。

统一规则：

- prediction只在计算指标和保存显示图时clamp到`[0,1]`；
- target不做额外后处理；
- 每张图在RGB三通道上独立计算PSNR和SSIM，再对图像求均值；
- 去噪先对三个sigma等权平均；
- 三任务macro再对denoise、derain和dehaze等权平均；
- 不按三个数据集的图像数量加权。

```text
denoise_metric = mean(sigma15, sigma25, sigma50)
macro_metric = mean(denoise_metric, derain_metric, dehaze_metric)
```

最佳checkpoint选择指标：

```text
val/macro/psnr
```

SSIM为辅助指标，不能改变checkpoint选择。

## 11. 正式测试

| 任务 | 测试集 | 数量 |
|---|---|---:|
| denoise | BSD68 × sigma15/25/50 | 204 |
| derain | Rain100L | 100 |
| dehaze | SOTS Outdoor | 500 |
| 总计 | AIO3-v1 | 804 |

测试固定为原始分辨率、batch size 1、无resize、无随机增强、无TTA、无tile。只有整图推理
确实无法运行时才允许启用tile；一旦启用，所有待比较模型必须统一tile size、overlap和
融合方法，并升级或补充协议标识。

参考runner命令：

```bash
python -m aio3_runner.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/best_macro_psnr.pth" \
  --num-workers 4
```

正式测试必须输出：

```text
test/state.json                         # completed, 804/804
test/metrics.json
test/metrics.csv
test/per_image_metrics.csv              # 805行，含表头
test/predictions/                        # 804张
test/gallery/                            # 14样本 × 5种图 = 70张PNG
```

测试程序必须拒绝非formal checkpoint、未完成run、脏Git工作区、commit不一致、manifest
不一致及覆盖已有测试输出。测试失败后不要静默删除输出并重跑，应先保存state和错误日志。

## 12. W&B统一使用方法

### 12.1 环境与组织

Python 3.9环境固定使用：

```text
wandb==0.25.1
project=aio3-restoration
group=aio3-v1
```

entity由项目所有者指定并写入config。当前参考entity为`c14150591-sjtu`。每个模型、seed、
run kind对应一个独立W&B run，推荐命名：

```text
<model_id>-<smoke|pilot|formal>-seed<seed>-<UTC timestamp>
```

服务器只需登录一次：

```bash
wandb login
python -c 'import wandb; print(wandb.__version__)'
```

W&B run、cache和staging目录必须位于`OUTPUT_ROOT`，不能写入源码或原始数据目录。

### 12.2 统一global step

所有日志字典必须带同一个`global_step`。所有标量和媒体横轴必须显式绑定到它：

```python
run.define_metric("global_step")
run.define_metric("train/*", step_metric="global_step")
run.define_metric("diagnostics/*", step_metric="global_step")
run.define_metric("system/*", step_metric="global_step")
run.define_metric("val/*", step_metric="global_step")
run.define_metric("test/*", step_metric="global_step")
```

`wandb==0.25.1`的单星号通配不能可靠覆盖多层路径，因此还必须对
`val/denoise/sigma50/psnr`等所有多层指标精确调用`define_metric`。网页横轴显示为W&B内部
`Step`而不是`global_step`，即视为监控验收失败。

### 12.3 每50 step训练标量

```text
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

每个点是最近50个optimizer steps的样本加权均值，而不是最后一个batch。三个
`train/samples_*`必须始终为`200/200/200`。

### 12.4 有符号残差诊断

只要模型可以定义`restored_raw - degraded`，就统一记录：

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
diagnostics/<task>/residual_negative_fraction
```

正负比例是诊断信号，不是优化目标。非零残差中某一符号长期精确为0、near-zero长期接近1
或prediction越界比例持续异常增长时才需要检查输出激活和训练稳定性。

### 12.5 每5000 step验证指标

```text
val/denoise/sigma15/{psnr,ssim}
val/denoise/sigma25/{psnr,ssim}
val/denoise/sigma50/{psnr,ssim}
val/denoise/mean/{psnr,ssim}
val/derain/{psnr,ssim}
val/dehaze/{psnr,ssim}
val/macro/{psnr,ssim}
```

同时更新：

```text
best/val_macro_psnr
best/val_macro_ssim
best/global_step
```

网页平滑只用于观察，不能用平滑后的极值选择checkpoint。

### 12.6 每10000 step固定可视化

`visual_samples.json`冻结14个验证样本：6个denoise、4个derain、4个dehaze。每个样本在
`val/fixed_samples`中固定展示：

```text
input, prediction, target, absolute_error, signed_residual,
psnr, ssim, residual_mean, residual_negative_fraction
```

同一sample ID在所有step的位置必须一致。absolute error固定显示`[0,0.25]`，signed
residual固定显示`[-0.25,0.25]`，禁止自动缩放色阶制造视觉改善。

### 12.7 推荐Workspace布局

1. **Training health**：总L1、三任务L1、学习率和梯度范数；
2. **Task balance**：三个samples计数；
3. **Validation**：三档去噪、去雨、去雾及macro；
4. **Signed residual diagnostics**：残差、越界比例和histogram；
5. **Qualitative restoration**：固定样本Table；
6. **Performance**：显存、step time、吞吐量和GPU utilization。

### 12.8 W&B不是唯一事实源

训练必须同时写本地JSONL和状态文件。网页暂时不刷新时先检查：

```bash
pgrep -af 'python.*aio3_runner.train'
cat "${RUN_DIR}/run_state.json"
tail -n 1 "${RUN_DIR}/train_metrics.jsonl" | python -m json.tool
cat "${RUN_DIR}/wandb_state.json"
```

W&B网络错误不得改变梯度和RNG。单次上传失败写入`logs/wandb_errors.jsonl`，本地训练继续。
不得仅因网页停止刷新就启动第二个训练进程。

### 12.9 Artifact

- 启动时上传manifest、审计、协议和config描述，不上传原始数据；
- 训练结束上传最佳checkpoint及复现metadata；
- 测试结束上传数值表和14样本gallery；
- 804张完整prediction只保存在服务器，不全部上传W&B。

## 13. 输出目录契约

```text
${OUTPUT_ROOT}/<model_id>/<run_name>/
├── config.yaml
├── environment.json
├── run_state.json
├── wandb_state.json
├── wandb_run_id.txt
├── train_metrics.jsonl
├── validation_metrics.jsonl
├── manifests/
├── checkpoints/
├── validation/
├── logs/
├── wandb/
└── test/
```

`run_state.json`状态限定为`running`、`validating`、`paused`、`interrupted`、`failed`或
`completed`。正式结果只接受`completed/200000`。

## 14. 其他模型接入验收清单

正式训练前逐项确认：

- [ ] 模型adapter输入输出形状和任意分辨率测试通过；
- [ ] 输出在loss前未clamp，且不存在意外最终激活；
- [ ] manifest五个SHA256与第4节完全一致；
- [ ] 随机crop、增强、sigma和噪声可复现；
- [ ] 每batch严格为4/4/4；
- [ ] 恢复训练的下一批样本和学习率精确一致；
- [ ] 参数、梯度和loss均为finite；
- [ ] BF16 forward/backward通过，特殊FP32边界已记录；
- [ ] checkpoint round-trip后固定输入输出一致；
- [ ] 100-step smoke及W&B global-step横轴通过；
- [ ] 暂停/恢复沿用同一W&B run ID；
- [ ] 5000-step pilot完成且14样本无明显伪影；
- [ ] Git工作区干净，config和commit已冻结；
- [ ] formal从随机初始化重新开始；
- [ ] 正式测试只使用best macro PSNR checkpoint；
- [ ] 测试输出满足804 prediction、805行CSV和70张gallery PNG。

任一项失败时不得发布`AIO3-v1`正式对比结果。

## 15. 模型合规报告模板

每个模型完成后提交一份以下格式的Markdown或JSON：

```text
protocol: aio3-v1
model_id:
model_description:
model_repository:
model_commit:
runner_commit:
seed:
parameters_total:
parameters_trainable:
precision:
special_fp32_boundaries:
pretrained: false
ema: false
tta: false
manifest_sha256:
run_dir:
wandb_url:
best_validation_step:
best_validation_macro_psnr:
best_validation_macro_ssim:
best_checkpoint_sha256:
test_bsd68_sigma15_psnr_ssim:
test_bsd68_sigma25_psnr_ssim:
test_bsd68_sigma50_psnr_ssim:
test_bsd68_mean_psnr_ssim:
test_rain100l_psnr_ssim:
test_sots_outdoor_psnr_ssim:
test_macro_psnr_ssim:
mean_inference_seconds:
test_predictions_count: 804
test_per_image_csv_lines: 805
test_gallery_png_count: 70
known_limitations:
```

## 16. 已验证DCNv4参考基线

以下结果用于检查其他实现是否遵循相同统计口径，不是调参目标。

```text
model_id: dcnv4_unet
seed: 3407
model parameters: 29,924,411
training/runner commit: 6a2c14f92770506fe3b2558ec4072037189b1ea9
best validation step: 145000
best validation macro PSNR/SSIM: 30.362972 / 0.907675
best checkpoint SHA256: 457810a9706094c9ac63ed492057c63c5f89d5bd08ed1baa5898ec52a8a94a8f
W&B: https://wandb.ai/c14150591-sjtu/aio3-restoration/runs/db94291336a148a0
```

| 测试项 | PSNR | SSIM |
|---|---:|---:|
| BSD68 sigma15 | 32.923013 | 0.898737 |
| BSD68 sigma25 | 30.376029 | 0.849709 |
| BSD68 sigma50 | 26.618784 | 0.692946 |
| BSD68 mean | 29.972609 | 0.813797 |
| Rain100L | 31.653237 | 0.929961 |
| SOTS Outdoor | 27.010384 | 0.951769 |
| **AIO3 task macro** | **29.545410** | **0.898509** |

评测metadata：原始分辨率、batch size 1、BF16网络/FP32 DCNv4、无TTA、无tile，共804张。

## 17. 协议变更

下列任一变化都不能继续使用`AIO3-v1`标签：

- 重新划分scene或改变manifest；
- 使用WED/noisy或改变高斯噪声集合/seed规则；
- 改变任务采样比例、有效batch、patch size或增强；
- 改变loss、optimizer、scheduler或训练步数；
- 添加预训练、EMA、TTA或模型专属监督；
- 改变PSNR/SSIM、clamp、裁边或宏平均方法；
- 改变测试pair、tile策略或使用测试集选择checkpoint。

纯工程修复若不改变数据、梯度和指标数值，可以保留协议版本，但必须记录修复commit并
重新运行受影响的验收。任何影响数学行为的修改都必须从头训练，并使用新的实验标签或
协议版本。
