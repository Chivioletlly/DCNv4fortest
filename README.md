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
