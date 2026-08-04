"""
Simple folder-based dataset for GeneralDecompositionNet.

Directory layout (any combination is valid):

    input_dir/          ← degraded images          (required)
    bg_dir/             ← clean background GT       (optional, maps to 'lol_gt')
    pattern_dir/        ← degradation pattern GT    (optional, maps to 'flare_gt')

Files are matched by filename stem (e.g. "0001.png" ↔ "0001.png").
If a GT directory is not provided, that key is simply absent from the batch.

Returned batch keys (compatible with train_general_decomp / inference_general_decomp):
    'input_image'  [B, 3, H, W]  float32 in [0, 1]
    'lol_gt'       [B, 3, H, W]  (only when bg_dir is given)
    'flare_gt'     [B, 3, H, W]  (only when pattern_dir is given)
    'filename'     list[str]     bare filename for reference
"""

import os
import glob
from typing import Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate
import torchvision.transforms.functional as TF
import random


_IMAGE_EXTS = ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp')


def _glob_images(directory: str):
    paths = set()
    for ext in _IMAGE_EXTS:
        paths.update(glob.glob(os.path.join(directory, f'*.{ext}')))
        paths.update(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    return sorted(list(paths))


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


class GeneralDecompDataset(Dataset):
    """
    Args:
        input_dir        : folder containing subfolders for each degradation type (required).
                           e.g., input_dir/blur/, input_dir/rain/, input_dir/noise/
        bg_dir           : folder of clean background GTs  (optional).
        pattern_dir      : folder containing subfolders for each degradation type's pattern GTs (optional).
                           e.g., pattern_dir/blur/, pattern_dir/rain/, pattern_dir/noise/
        degradation_types: list of degradation type labels (e.g., ['blur', 'rain', 'noise']).
                           If None, will auto-detect from input_dir subfolders.
        sample_stems_dir  : optional image directory whose filename stems select a split.
                            Images are still loaded from input_dir at native resolution.
        height, width    : legacy resize target. If both are None, images keep their size.
        patch_size       : synchronized random crop size for progressive training.
                           Mutually exclusive with height/width.
        augment          : enable random horizontal flip during training.
    """

    def __init__(
        self,
        input_dir: str,
        bg_dir: Optional[str] = None,
        pattern_dir: Optional[str] = None,
        degradation_types: Optional[list] = None,
        sample_stems_dir: Optional[str] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        patch_size: Optional[int] = None,
        augment: bool = False,
    ):
        if patch_size is not None and (height is not None or width is not None):
            raise ValueError('patch_size is mutually exclusive with height/width resizing')
        if (height is None) != (width is None):
            raise ValueError('height and width must either both be set or both be None')
        self.height   = height
        self.width    = width
        self.patch_size = None
        self.set_patch_size(patch_size)
        self.augment  = augment

        # ---------- auto-detect degradation types ----------
        root_input_paths = _glob_images(input_dir)
        if degradation_types is None:
            degradation_types = sorted([
                d for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d))
            ])
            if not degradation_types and root_input_paths:
                degradation_types = ['input']
            elif not degradation_types:
                raise ValueError(f'No image files or subfolders found in input_dir: {input_dir}')

        self.degradation_types = degradation_types
        selected_stems = None
        if sample_stems_dir:
            selected_stems = {_stem(path) for path in _glob_images(sample_stems_dir)}
            if not selected_stems:
                raise FileNotFoundError(
                    f'No split-selector images found in {sample_stems_dir}'
                )

        # ---------- build samples for each degradation type ----------
        self.samples = []
        matched_stems = set()
        bg_map = _build_stem_map(bg_dir) if bg_dir else {}
        for deg_type in self.degradation_types:
            # input images: input_dir/deg_type/*.png
            deg_input_dir = os.path.join(input_dir, deg_type)
            if os.path.isdir(deg_input_dir):
                input_paths = _glob_images(deg_input_dir)
            elif len(self.degradation_types) == 1 and root_input_paths:
                # Also accept a flat image directory for inference and one-type datasets.
                deg_input_dir = input_dir
                input_paths = root_input_paths
            else:
                raise FileNotFoundError(f'Degradation folder not found: {deg_input_dir}')
            if len(input_paths) == 0:
                raise FileNotFoundError(f'No images found in {deg_input_dir}')
            if selected_stems is not None:
                input_paths = [path for path in input_paths if _stem(path) in selected_stems]

            # pattern GT: pattern_dir/deg_type/*.png
            deg_pattern_dir = os.path.join(pattern_dir, deg_type) if pattern_dir else None
            pattern_map = _build_stem_map(deg_pattern_dir) if deg_pattern_dir else {}

            # build samples for this degradation type
            for p in input_paths:
                s = _stem(p)
                bg_path = bg_map.get(s)
                pattern_path = pattern_map.get(s)

                # if a GT dir is specified but this file has no match, skip it
                if bg_dir and bg_path is None: continue
                if pattern_dir and pattern_path is None: continue

                self.samples.append({
                    'input':         p,
                    'bg':            bg_path,
                    'pattern':       pattern_path,
                    'stem':          s,
                    'deg_type':      deg_type,
                })
                matched_stems.add(s)

        if selected_stems is not None:
            missing_stems = sorted(selected_stems - matched_stems)
            if missing_stems:
                preview = ', '.join(missing_stems[:10])
                raise ValueError(
                    f'{len(missing_stems)} selected stems have no complete original pair; '
                    f'first missing: {preview}'
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                'Dataset is empty after GT matching. '
                'Check that filenames in input_dir/*/ , bg_dir, pattern_dir/*/ share the same stems.'
            )

    def __len__(self):
        return len(self.samples)

    def set_patch_size(self, patch_size: Optional[int]):
        if patch_size is not None and patch_size <= 0:
            raise ValueError(f'patch_size must be positive, got {patch_size}')
        self.patch_size = patch_size

    def __getitem__(self, idx):
        info = self.samples[idx]

        inp  = _load_rgb(info['input'])
        bg   = _load_rgb(info['bg'])      if info['bg']      else None
        pat  = _load_rgb(info['pattern']) if info['pattern'] else None

        reference_size = inp.size
        for name, image in (('background', bg), ('pattern', pat)):
            if image is not None and image.size != reference_size:
                raise ValueError(
                    f"Spatial size mismatch for {info['stem']}: input={reference_size}, "
                    f'{name}={image.size}'
                )

        # ----- resize -----
        if self.height is not None and self.width is not None:
            inp = TF.resize(inp, [self.height, self.width], interpolation=Image.BICUBIC)
            if bg  is not None: bg  = TF.resize(bg,  [self.height, self.width], interpolation=Image.BICUBIC)
            if pat is not None: pat = TF.resize(pat, [self.height, self.width], interpolation=Image.BICUBIC)

        # ----- to tensor [0, 1] -----
        inp_t = TF.to_tensor(inp)
        bg_t = TF.to_tensor(bg) if bg is not None else None
        pat_t = TF.to_tensor(pat) if pat is not None else None

        # ----- augmentation (synchronized across input and all targets) -----
        if self.augment and random.random() < 0.5:
            inp_t = TF.hflip(inp_t)
            if bg_t is not None: bg_t = TF.hflip(bg_t)
            if pat_t is not None: pat_t = TF.hflip(pat_t)

        if self.patch_size is not None:
            tensors = [inp_t, bg_t, pat_t]
            tensors = _synchronized_random_crop(tensors, self.patch_size)
            inp_t, bg_t, pat_t = tensors

        batch = {
            'input_image':       inp_t,
            'degradation_type':  info['deg_type'],
            'filename':           info['stem'],
        }
        if bg_t  is not None: batch['lol_gt']   = bg_t
        if pat_t is not None: batch['flare_gt'] = pat_t

        return batch


def _build_stem_map(directory: str) -> dict:
    return {_stem(p): p for p in _glob_images(directory)}


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert('RGB')


def _pad_to_minimum(tensor: torch.Tensor, size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    missing_height = max(0, size - height)
    missing_width = max(0, size - width)
    left = missing_width // 2
    right = missing_width - left
    top = missing_height // 2
    bottom = missing_height - top
    if not any((left, right, top, bottom)):
        return tensor
    mode = (
        'reflect'
        if max(top, bottom) < height and max(left, right) < width
        else 'replicate'
    )
    return F.pad(tensor, (left, right, top, bottom), mode=mode)


def _synchronized_random_crop(tensors, patch_size: int):
    padded = [
        _pad_to_minimum(tensor, patch_size) if tensor is not None else None
        for tensor in tensors
    ]
    height, width = padded[0].shape[-2:]
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    return [
        tensor[..., top:top + patch_size, left:left + patch_size]
        if tensor is not None else None
        for tensor in padded
    ]


def collate_same_size(batch):
    """Fail clearly when native-resolution images of different sizes share a batch."""
    shapes = {tuple(sample['input_image'].shape) for sample in batch}
    if len(shapes) > 1:
        raise ValueError(
            'A native-resolution batch contains different image sizes. '
            'Use batch_size=1, fixed resizing, or patch_size training.'
        )
    return default_collate(batch)

def build_dataloader(
    input_dir: str,
    bg_dir: Optional[str] = None,
    pattern_dir: Optional[str] = None,
    degradation_types: Optional[list] = None,
    sample_stems_dir: Optional[str] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    patch_size: Optional[int] = None,
    batch_size: int = 8,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, GeneralDecompDataset]:
    ds = GeneralDecompDataset(
        input_dir=input_dir,
        bg_dir=bg_dir,
        pattern_dir=pattern_dir,
        degradation_types=degradation_types,
        sample_stems_dir=sample_stems_dir,
        height=height,
        width=width,
        patch_size=patch_size,
        augment=augment,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_same_size,
    )
    return loader, ds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',   required=True)
    parser.add_argument('--bg_dir',      default=None)
    parser.add_argument('--pattern_dir', default=None)
    parser.add_argument('--height',      type=int, default=256)
    parser.add_argument('--width',       type=int, default=256)
    args = parser.parse_args()

    loader, ds = build_dataloader(
        input_dir=args.input_dir,
        bg_dir=args.bg_dir,
        pattern_dir=args.pattern_dir,
        height=args.height,
        width=args.width,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print(f'Dataset size: {len(ds)}')
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape}  range [{v.min():.3f}, {v.max():.3f}]')
        else:
            print(f'  {k}: {v}')
