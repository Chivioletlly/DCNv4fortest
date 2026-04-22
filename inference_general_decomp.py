"""
Inference script for GeneralDecompositionNet.

Outputs per image:
    pattern       – predicted degradation component
    background    – predicted clean background
    reconstructed – pattern + background (additive reconstruction)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from general_decomp.general_decomposition_model import GeneralDecompositionNet
from general_decomp.dataset import build_dataloader


class GeneralDecompositionInference:

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt.get('config', {})

        self.model = GeneralDecompositionNet(
            in_channels=self.config.get('in_channels', 3),
            base_channels=self.config.get('base_channels', 64),
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        print(f'Model loaded from {checkpoint_path}')
        print(f'Config: in_channels={self.config.get("in_channels",3)}, '
              f'base_channels={self.config.get("base_channels",64)}')

    # ------------------------------------------------------------------
    # Dataset-level inference
    # ------------------------------------------------------------------

    def create_dataloader(self, input_dir: str, batch_size: int = 1):
        loader, ds = build_dataloader(
            input_dir=input_dir,
            height=self.config.get('image_height', 512),
            width=self.config.get('image_width', 512),
            batch_size=batch_size,
            augment=False,
            shuffle=False,
            num_workers=0,
        )
        print(f'Dataset size: {len(ds)}')
        return loader

    def inference_batch(self, dataloader, num_samples=None, save_dir='./inference_results'):
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_samples is not None and i >= num_samples:
                    break

                inp = batch['input_image'].to(self.device)
                pattern, background, _ = self.model(inp)
                reconstructed = (pattern + background).clamp(0, 1)

                B = inp.shape[0]
                # Get degradation types from batch (as list)
                deg_types = batch.get('degradation_type', [None] * B)
                filenames = batch.get('filename', [None] * B)

                for j in range(B):
                    grid = vutils.make_grid(
                        torch.cat([
                            inp[j:j+1],
                            pattern[j:j+1],
                            background[j:j+1],
                            reconstructed[j:j+1],
                        ]),
                        nrow=4, normalize=True, padding=2,
                    )
                    idx = i * B + j
                    deg_type = deg_types[j] if isinstance(deg_types, list) else deg_types
                    fname = filenames[j] if isinstance(filenames, list) else filenames
                    # Build filename: inference_{idx:04d}_{deg_type}_{stem}.png
                    deg_suffix = f'_{deg_type}' if deg_type else ''
                    stem_suffix = f'_{fname}' if fname else ''
                    out_path = os.path.join(save_dir, f'inference_{deg_suffix}{stem_suffix}.png')
                    vutils.save_image(grid, out_path)
                    print(f'Saved: {out_path}')
                    self._display(grid, f'{deg_suffix}')

    # ------------------------------------------------------------------
    # Single-image inference
    # ------------------------------------------------------------------

    def inference_single_image(self, image_path: str, save_path: str = None):
        from torchvision import transforms

        h = self.config.get('image_height', 512)
        w = self.config.get('image_width',  512)
        tf = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])

        img = Image.open(image_path).convert('RGB')
        inp = tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pattern, background, _ = self.model(inp)
            reconstructed = (pattern + background).clamp(0, 1)

        grid = vutils.make_grid(
            torch.cat([inp, pattern, background, reconstructed]),
            nrow=4, normalize=True, padding=2,
        )

        if save_path:
            vutils.save_image(grid, save_path)
            print(f'Saved: {save_path}')

        self._display(grid, os.path.basename(image_path))

        return {
            'input':         inp,
            'pattern':       pattern,
            'background':    background,
            'reconstructed': reconstructed,
            'grid':          grid,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _display(grid_tensor, title):
        grid_np = grid_tensor.cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(16, 4))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(f'{title}  |  Input  |  Pattern  |  Background  |  Reconstructed')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='General Decomposition Inference')

    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--input_dir',   type=str, default=None,
                        help='Folder of degraded images for batch inference')
    parser.add_argument('--image_path',  type=str, default=None,
                        help='Single image path for single-image inference')
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max number of samples to process (default: all)')
    parser.add_argument('--save_dir',    type=str,
                        default='./inference_results/general_decomp')
    parser.add_argument('--device',      type=str, default='cuda')

    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = GeneralDecompositionInference(args.checkpoint, args.device)

    if args.input_dir:
        loader = inferencer.create_dataloader(args.input_dir, args.batch_size)
        inferencer.inference_batch(loader, args.num_samples, args.save_dir)
        print(f'Results saved to: {args.save_dir}')

    elif args.image_path:
        stem = os.path.splitext(os.path.basename(args.image_path))[0]
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f'{stem}_decomp.png')
        inferencer.inference_single_image(args.image_path, save_path)

    else:
        print('Error: specify --input_dir or --image_path')


if __name__ == '__main__':
    main()
