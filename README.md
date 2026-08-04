# General Degradation Decomposition Network

General Degradation Decomposition Network for image processing and restoration tasks.

## Installation

```bash
uv sync
```

## DCNv4 on the Linux training server

The optional `--use_orient_block` path uses the official DCNv4 CUDA operator vendored
from OpenGVLab at commit `4b848f7dd7da74ff03f7d278f902c6fd05b391b5`.
The tested build target is:

- Linux
- PyTorch 2.5.1 with CUDA 12.4
- CUDA Toolkit / `nvcc` 12.4
- `CUDA_HOME` pointing to the CUDA 12.4 toolkit

After installing the project dependencies, compile and smoke-test the extension:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
bash scripts/build_dcnv4.sh
```

Run the full 512x512, two-step integration test and the legacy-vs-DCNv4 benchmark:

```bash
python scripts/test_dcnv4.py --resolution 512 --steps 2
python scripts/benchmark_dcnv4.py --resolution 512 --warmup 20 --iterations 100
```

Enable all four residual DCNv4 blocks during training with:

```bash
python train_general_decomp.py \
  --input_dir /path/to/degraded \
  --bg_dir /path/to/clean \
  --use_orient_block
```

DCNv4 checkpoints use `architecture_version=2` and
`direction_block_type=dcnv4`. Checkpoints trained with the removed directional-strip
block are intentionally rejected; train a new DCNv4 model instead of resuming them.

## Native-resolution and progressive-patch training

The model accepts arbitrary image height and width. Like Restormer, inputs are
reflection-padded on the bottom/right to the model's size multiple (8) and predictions
are cropped back to the original size. Model parameters and checkpoint keys are
unchanged, so existing DCNv4 checkpoints remain load-compatible.

For training, `--patch_schedule` reads original images without resizing and applies the
same random crop to the degraded input and all GT images. The patch size changes at the
specified one-indexed epochs. Validation automatically runs at native resolution when a
patch schedule is enabled and therefore requires `--val_batch_size 1`.

The following command preserves the existing Rain13K split by using images in the old
512 views only as filename-stem selectors; pixels are loaded from the original dataset:

```bash
RAW_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/train/Rain13K
SPLIT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512
RUN_DIR=./checkpoints/rain13k_unet_dcnv4_native_progressive_v1

python train_general_decomp.py \
  --input_dir "$RAW_ROOT" \
  --bg_dir "$RAW_ROOT/target" \
  --degradation_types input \
  --train_stems_dir "$SPLIT_ROOT/unet_train/input" \
  --val_input_dir "$RAW_ROOT" \
  --val_bg_dir "$RAW_ROOT/target" \
  --val_degradation_types input \
  --val_stems_dir "$SPLIT_ROOT/validation/input" \
  --epochs 100 \
  --batch_size 16 \
  --val_batch_size 1 \
  --patch_schedule 1:128 11:160 26:192 41:256 61:320 81:384 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --min_lr 1e-7 \
  --grad_clip 1.0 \
  --w_orthogonal 0.0 \
  --w_pattern 1.0 \
  --w_bg 1.0 \
  --w_recon 0.0 \
  --w_ssim 0.5 \
  --w_frequency 0.2 \
  --w_edge 0.2 \
  --base_channels 64 \
  --bottleneck_type conv \
  --use_orient_block \
  --num_workers 4 \
  --save_interval 10 \
  --val_interval 5 \
  --checkpoint_dir "$RUN_DIR"
```

Use `--resume_from "$RUN_DIR/latest.pth"` only when continuing the same schedule.
Use `--init_from /path/to/old_checkpoint.pth` to initialize model weights while resetting
the optimizer, scheduler, epoch, and best metric for a new schedule.

Native-size inference is now the default and mixed-size folders should use batch size 1:

```bash
python inference_general_decomp.py \
  --checkpoint /path/to/best.pth \
  --input_dir /path/to/input \
  --batch_size 1 \
  --save_dir ./inference_results/native
```

Add `--resize_to_checkpoint` only to reproduce the legacy fixed-size inference behavior.

## TensorBoard metrics

Mixed-degradation batches are logged with stable, sample-level groups instead of using
the Python representation of the whole batch as a tag:

- `train_batch/overall/*` and `train_epoch/overall/*` contain the real optimization
  losses for the complete mixed batch or epoch.
- `train_batch/by_type/<type>/*` and `train_epoch/by_type/<type>/*` contain exact
  per-sample image losses for each degradation type. `image_total` intentionally excludes
  the batch-level orthogonality term, which cannot be attributed to one type.
- `val_samples/sample_XX/*` keeps stable image tags across epochs, so TensorBoard shows a
  time series instead of creating a new panel for every epoch.

Epoch metrics are weighted by sample count, including the final partial batch. Resuming
from a checkpoint continues at the next epoch and restores the best validation loss.

## Dependencies

- torch == 2.5.1 (CUDA 12.4 build on the training server)
- torchvision == 0.20.1
- einops >= 0.7.0
- pillow >= 9.0.0
- tqdm >= 4.65.0
- ninja >= 1.11.0

## License

MIT
