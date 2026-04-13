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
from torch.utils.data import Dataset, DataLoader
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
        height, width    : resize target. If both are None, images are not resized.
        augment          : enable random horizontal flip during training.
    """

    def __init__(
        self,
        input_dir: str,
        bg_dir: Optional[str] = None,
        pattern_dir: Optional[str] = None,
        degradation_types: Optional[list] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        augment: bool = False,
    ):
        self.height   = height
        self.width    = width
        self.augment  = augment

        # ---------- auto-detect degradation types ----------
        if degradation_types is None:
            degradation_types = sorted([
                d for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d))
            ])
            if not degradation_types:
                raise ValueError(f'No subfolders found in input_dir: {input_dir}')

        self.degradation_types = degradation_types

        # ---------- build samples for each degradation type ----------
        self.samples = []
        for deg_type in self.degradation_types:
            # input images: input_dir/deg_type/*.png
            deg_input_dir = os.path.join(input_dir, deg_type)
            if not os.path.exists(deg_input_dir):
                raise FileNotFoundError(f'Degradation folder not found: {deg_input_dir}')
            input_paths = _glob_images(deg_input_dir)
            if len(input_paths) == 0:
                raise FileNotFoundError(f'No images found in {deg_input_dir}')

            # bg GT: bg_dir/*.png (shared across degradation types)
            bg_map = _build_stem_map(bg_dir) if bg_dir else {}

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

        if len(self.samples) == 0:
            raise RuntimeError(
                'Dataset is empty after GT matching. '
                'Check that filenames in input_dir/*/ , bg_dir, pattern_dir/*/ share the same stems.'
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        inp  = _load_rgb(info['input'])
        bg   = _load_rgb(info['bg'])      if info['bg']      else None
        pat  = _load_rgb(info['pattern']) if info['pattern'] else None

        # ----- resize -----
        if self.height is not None and self.width is not None:
            inp = TF.resize(inp, [self.height, self.width], interpolation=Image.BICUBIC)
            if bg  is not None: bg  = TF.resize(bg,  [self.height, self.width], interpolation=Image.BICUBIC)
            if pat is not None: pat = TF.resize(pat, [self.height, self.width], interpolation=Image.BICUBIC)

        # ----- augmentation (same random state for all images in a sample) -----
        if self.augment and random.random() < 0.5:
            inp = TF.hflip(inp)
            if bg  is not None: bg  = TF.hflip(bg)
            if pat is not None: pat = TF.hflip(pat)

        # ----- to tensor [0, 1] -----
        inp_t = TF.to_tensor(inp)
        batch = {
            'input_image':       inp_t,
            'degradation_type':  info['deg_type'],
            'filename':           info['stem'],
        }
        if bg  is not None: batch['lol_gt']   = TF.to_tensor(bg)
        if pat is not None: batch['flare_gt']  = TF.to_tensor(pat)

        return batch


def _build_stem_map(directory: str) -> dict:
    return {_stem(p): p for p in _glob_images(directory)}


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')

def build_dataloader(
    input_dir: str,
    bg_dir: Optional[str] = None,
    pattern_dir: Optional[str] = None,
    degradation_types: Optional[list] = None,
    height: int = 512,
    width: int = 512,
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
        height=height,
        width=width,
        augment=augment,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
