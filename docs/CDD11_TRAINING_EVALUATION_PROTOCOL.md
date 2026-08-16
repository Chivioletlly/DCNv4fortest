# CDD-11 三网络训练与测试规范（cdd11-v1）

本文件是 `unet`、`degradation_aware_unet` 和 `uformer` 在 CDD-11 上进行公平比较的冻结协议。正式结果不得在三种模型之间改变数据划分、采样、损失、优化器、训练步数、验证指标或推理模式。

## 1. 数据布局与划分

数据根目录必须具有以下结构，并且 `clear` 与 11 个退化目录中的文件名逐一对应：

```text
CDD-11/
  train/
    clear/ low/ haze/ rain/ snow/ low_haze/ low_rain/ low_snow/
    haze_rain/ haze_snow/ low_haze_rain/ low_haze_snow/
  test/
    clear/ low/ haze/ rain/ snow/ low_haze/ low_rain/ low_snow/
    haze_rain/ haze_snow/ low_haze_rain/ low_haze_snow/
```

正式审计要求：训练源 1183 个清晰场景、测试源 200 个清晰场景、每张图为 1080×720 RGB PNG。训练源按场景划分为 1083 个训练场景和 100 个验证场景；验证场景是按 `SHA256("cdd11-v1:split:3407:<filename>")` 排序后的前 100 个。官方测试集不参与模型选择。

生成并审计清单：

```powershell
python -m cdd11_runner.prepare_data `
  --data-root D:\datasets\CDD-11 `
  --output-dir D:\experiments\cdd11-v1-manifests
```

命令会生成 `train.jsonl`（11913 行）、`val.jsonl`（1100 行）、`test.jsonl`（2200 行）、`data_audit.json` 和固定验证可视化清单。任何数量、文件名、尺寸、颜色模式、跨官方划分的清晰图精确重复或哈希不一致都会终止运行。

## 2. 模型身份

| CLI 模型 ID | 固定结构 | 预期参数量 |
|---|---|---:|
| `unet` | DCNv4 U-Net，APG-000 | 29,924,411 |
| `degradation_aware_unet` | 退化感知 DCNv4 U-Net，APG-111 | 34,852,539 |
| `uformer` | Uformer-B | 50,880,946 |

所有模型均从随机初始化训练，不使用预训练权重、EMA 或退化标签输入。Uformer 通过 `--uformer-root` 显式加载，其独立 Git 提交会写入配置和 checkpoint。创建实验前，主仓库和 Uformer 仓库必须是干净工作树。

运行 DCNv4 两个模型前还必须按仓库现有说明编译 vendored DCNv4 CUDA extension；若 Python 无法导入 `DCNv4`，训练会在创建模型时直接终止。

## 3. 公共训练设置

- patch：256×256，同步随机裁剪、水平/垂直翻转与 90° 旋转。
- 一个有效 batch：11 个样本，每种退化恰好 1 个；类别内先均匀选场景，再选样本。
- 微批：所有模型使用同一个 `--microbatch-size`。每个微批损失为 `sum(per_sample_L1) / 11`，完成全部微批后只进行一次梯度裁剪、一次 `optimizer.step()` 和一次 `scheduler.step()`。
- 精度：FP32 参数、BF16 autocast；损失前不裁剪输出。
- 损失：RGB L1。
- 优化器：AdamW，学习率 `2e-4`，betas `(0.9, 0.999)`，weight decay `1e-4`。
- 调度：线性 warmup 2000 optimizer steps，随后 cosine 衰减到 `1e-6`。
- 梯度裁剪：global norm 1.0。
- 正式训练：200000 optimizer steps；每 5000 步验证并保存 checkpoint，每 50000 步保存里程碑。
- 随机种子：3407。采样由全局 optimizer step 决定，因此恢复训练不会重复或跳过 batch。

## 4. 验收阶段

按模型逐一执行，但三个模型必须使用同一清单、种子、微批大小和推理模式。

1. `smoke`：100 步。先使用 `--pause-at-step 50`，再从 `latest.pth` 恢复到 100 步，验证 checkpoint、RNG、采样器和调度器恢复。
2. `pilot`：5000 步。用于检测显存、速度、数值稳定性及原分辨率验证是否可行。
3. `formal`：200000 步。只有 smoke 和 pilot 全部通过后才能启动。

每个阶段创建完三个运行后，先执行跨模型公平性审计。命令必须各提供一次三个模型的运行目录；它会拒绝数据哈希、种子、微批、训练超参数或推理模式的任何漂移：

```powershell
python -m cdd11_runner.audit_suite `
  --run-dir D:\experiments\...\unet-run `
  --run-dir D:\experiments\...\aware-run `
  --run-dir D:\experiments\...\uformer-run `
  --output D:\experiments\...\comparison_audit.json
```

新建 smoke 示例：

```powershell
python -m cdd11_runner.train `
  --model unet `
  --manifest-dir D:\experiments\cdd11-v1-manifests `
  --output-root D:\experiments\cdd11-v1 `
  --run-kind smoke `
  --microbatch-size 1 `
  --inference-mode native `
  --pause-at-step 50
```

退化感知 U-Net 只需把模型改为 `degradation_aware_unet`。Uformer 还需增加：

```powershell
--model uformer --uformer-root C:\path\to\uformer4test
```

精确恢复只接受该运行目录中的 `checkpoints/latest.pth`：

```powershell
python -m cdd11_runner.train --resume D:\experiments\...\checkpoints\latest.pth
```

## 5. 验证与推理模式

默认使用 batch size 1 的原分辨率推理。若 pilot 中任一模型在原分辨率验证发生显存不足，则废弃三者的该轮 pilot，并对三个模型统一使用：

```powershell
--inference-mode tiled
```

共享 tiled 模式固定为 512×512 tile、128 像素重叠和 Hann 加权融合。不得只给某个模型使用 tiled 模式，也不得在训练结束后更改正式运行配置。无 TTA。

验证指标为预测裁剪到 `[0,1]` 后的 RGB PSNR 和 RGB SSIM。记录每一类指标、11 类等权宏平均，以及单重、双重、三重退化的类别宏平均。最优 checkpoint 仅按验证集 `macro/psnr` 选择。

## 6. 正式测试

只有状态为 `completed`、达到 200000 步的 formal 运行可以读取测试清单。使用该运行的验证集最优 checkpoint：

```powershell
python -m cdd11_runner.evaluate `
  --checkpoint D:\experiments\...\checkpoints\best_macro_psnr.pth `
  --num-workers 4
```

每个正式测试必须产生：

- 2200 张 PNG 预测图；
- `per_image_metrics.csv`，含表头共 2201 行；
- `metrics.json` 与 `metrics.csv`，包含 11 类、单/双/三重组和总体宏平均；
- 固定的 2 个测试场景 × 11 类，共 22 组 gallery 样本；
- checkpoint、清单、主仓库及 Uformer 仓库提交哈希和推理模式元数据。

正式测试产物禁止覆盖。若运行失败，保留 `test/state.json` 与已有文件进行诊断，修复后必须新建输出运行，不得手工拼接结果。
