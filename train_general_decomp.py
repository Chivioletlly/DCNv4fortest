import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import logging
import argparse
from tqdm import tqdm
from typing import Dict

from general_decomp.general_decomposition_model import (
    GeneralDecompositionNet, DecompositionLoss, count_parameters
)
from general_decomp.dataset import build_dataloader


class GeneralDecompositionTrainer:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.model = GeneralDecompositionNet(
            in_channels=config.get('in_channels', 3),
            base_channels=config.get('base_channels', 64),
        ).to(self.device)

        self.criterion = DecompositionLoss(
            w_orthogonal=config.get('w_orthogonal', 0.1),
            w_pattern=config.get('w_pattern', 1.0),
            w_bg=config.get('w_bg', 1.0),
            w_recon=config.get('w_recon', 1.0),
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200),
            eta_min=config.get('min_lr', 1e-7),
        )

        self.epochs        = config.get('epochs', 200)
        self.save_interval = config.get('save_interval', 20)
        self.log_interval  = config.get('log_interval', 10)
        self.val_interval  = config.get('val_interval', 5)

        self.checkpoint_dir  = config.get('checkpoint_dir', './checkpoints/general_decomp')
        self.tensorboard_dir = config.get('tensorboard_dir') or os.path.join(self.checkpoint_dir, 'tensorboard')
        os.makedirs(self.checkpoint_dir,  exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self._setup_logging()

        self.logger.info('Parameters: %s', count_parameters(self.model))
        self.writer.add_text('config', '\n'.join(f'{k}: {v}' for k, v in config.items()), 0)
        self.logger.info('TensorBoard: %s', self.tensorboard_dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logging(self):
        log_file = os.path.join(self.checkpoint_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _make_dataloader(self, input_dir, bg_dir, pattern_dir, degradation_types, shuffle):
        bs = self.config.get('batch_size', 8) if shuffle else self.config.get('val_batch_size', 4)
        loader, ds = build_dataloader(
            input_dir=input_dir,
            bg_dir=bg_dir,
            pattern_dir=pattern_dir,
            degradation_types=degradation_types,
            height=self.config.get('image_height', 512),
            width=self.config.get('image_width', 512),
            batch_size=bs,
            augment=shuffle,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
        )
        return loader, ds

    def create_dataloaders(self):
        self.train_loader, train_ds = self._make_dataloader(
            input_dir=self.config['input_dir'],
            bg_dir=self.config.get('bg_dir'),
            pattern_dir=self.config.get('pattern_dir'),
            degradation_types=self.config.get('degradation_types'),
            shuffle=True,
        )
        self.logger.info('Train set: %d samples (degradation types: %s)',
                         len(train_ds), train_ds.degradation_types)

        self.val_loader = None
        if self.config.get('val_input_dir'):
            self.val_loader, val_ds = self._make_dataloader(
                input_dir=self.config['val_input_dir'],
                bg_dir=self.config.get('val_bg_dir'),
                pattern_dir=self.config.get('val_pattern_dir'),
                degradation_types=self.config.get('val_degradation_types'),
                shuffle=False,
            )
            self.logger.info('Val set: %d samples (degradation types: %s)',
                             len(val_ds), val_ds.degradation_types)


    @staticmethod
    def _unpack_batch(batch, device):
        """Extract tensors from a dataloader batch."""
        inp = batch['input_image'].to(device)
        # lol_gt → clean background,  flare_gt → degradation pattern
        bg_gt      = batch['lol_gt'].to(device)   if 'lol_gt'   in batch else None
        pattern_gt = batch['flare_gt'].to(device)  if 'flare_gt' in batch else None
        deg_type   = batch['degradation_type'] if 'degradation_type' in batch else 'unknown'
        return inp, bg_gt, pattern_gt, deg_type

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def _forward_and_loss(self, inp, bg_gt, pattern_gt):
        pattern, background, orth_loss = self.model(inp)
        total_loss, loss_dict = self.criterion(
            pattern, background, orth_loss, inp, pattern_gt, bg_gt
        )
        return pattern, background, total_loss, loss_dict

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        epoch_losses: Dict = {}
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                inp, bg_gt, pattern_gt, deg_type = self._unpack_batch(batch, self.device)

                self.optimizer.zero_grad()
                pattern, background, total_loss, loss_dict = self._forward_and_loss(
                    inp, bg_gt, pattern_gt
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get('grad_clip', 1.0)
                )
                self.optimizer.step()

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        epoch_losses.setdefault(k, []).append(v.item())

                # 按退化类型聚合损失
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        epoch_losses.setdefault(f'{deg_type}/{k}', []).append(v.item())

                pbar.set_postfix({
                    'loss':  f'{total_loss.item():.4f}',
                    'recon': f'{loss_dict.get("recon_l2", 0):.4f}',
                    'orth':  f'{loss_dict.get("orthogonal", 0):.4f}',
                    'lr':    f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                })

                if batch_idx % self.log_interval == 0:
                    self._log_batch(epoch, batch_idx, loss_dict, num_batches, deg_type)

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    def validate(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses: Dict = {}
        recon_errors = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                inp, bg_gt, pattern_gt, deg_type = self._unpack_batch(batch, self.device)
                pattern, background, total_loss, loss_dict = self._forward_and_loss(
                    inp, bg_gt, pattern_gt
                )
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        val_losses.setdefault(k, []).append(v.item())
                        # 按退化类型聚合
                        val_losses.setdefault(f'{deg_type}/{k}', []).append(v.item())

                recon = (pattern + background).clamp(0, 1)
                recon_errors.append(torch.mean((recon - inp) ** 2).item())

        avg = {k: sum(v) / len(v) for k, v in val_losses.items()}
        avg['avg_recon_mse'] = sum(recon_errors) / len(recon_errors)
        return avg


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'latest.pth'))

        if (epoch + 1) % self.save_interval == 0:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, f'epoch_{epoch+1:03d}.pth'))

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(ckpt, best_path)
            self.logger.info('Best model saved → %s', best_path)

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            self.logger.warning('Checkpoint not found: %s', path)
            return 0
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch = ckpt.get('epoch', 0)
        self.logger.info('Resumed from epoch %d (%s)', epoch, path)
        return epoch



    def _log_batch(self, epoch, batch_idx, loss_dict, num_batches, deg_type='unknown'):
        step = epoch * num_batches + batch_idx
        # 总体损失（不区分退化类型）
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.writer.add_scalar(f'train/batch_{k}', v.item(), step)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)

        # 按退化类型记录
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.writer.add_scalar(f'train/{deg_type}/batch_{k}', v.item(), step)

        loss_str = ' | '.join(
            f'{k}: {v.item():.4f}' for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
        )
        self.logger.info('Epoch %3d | Batch %4d/%4d | [%s] %s', epoch + 1, batch_idx, num_batches, deg_type, loss_str)

    def _log_epoch(self, epoch, train_losses, val_losses):
        # 总体损失（不区分退化类型）
        overall_train = {k: v for k, v in train_losses.items() if '/' not in k}
        overall_val = {k: v for k, v in val_losses.items() if '/' not in k}

        self.logger.info('Epoch %3d Train | %s', epoch + 1,
                         ' | '.join(f'{k}: {v:.4f}' for k, v in overall_train.items()))
        if overall_val:
            self.logger.info('Epoch %3d Val   | %s', epoch + 1,
                             ' | '.join(f'{k}: {v:.4f}' for k, v in overall_val.items()))

        # 记录总体损失
        for k, v in overall_train.items():
            self.writer.add_scalar(f'train_epoch/{k}', v, epoch + 1)
        for k, v in overall_val.items():
            self.writer.add_scalar(f'val_epoch/{k}', v, epoch + 1)

        # 按退化类型记录
        for k, v in train_losses.items():
            if '/' in k:
                self.writer.add_scalar(f'train_epoch/{k}', v, epoch + 1)
        for k, v in val_losses.items():
            if '/' in k:
                self.writer.add_scalar(f'val_epoch/{k}', v, epoch + 1)

        self.writer.flush()

    def save_sample_results(self, epoch: int, num_samples: int = 4):
        if self.val_loader is None:
            return

        self.model.eval()
        save_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                inp, bg_gt, pattern_gt, deg_type = self._unpack_batch(batch, self.device)
                pattern, background, _, _ = self._forward_and_loss(inp, bg_gt, pattern_gt)
                reconstructed = (pattern + background).clamp(0, 1)

                # Build visualisation row: input | pattern | bg | reconstructed [| GT bg | GT pattern]
                imgs = [inp[0:1], pattern[0:1], background[0:1], reconstructed[0:1]]
                labels = ['Input', 'Pattern', 'Background', 'Reconstructed']

                if bg_gt is not None:
                    imgs.append(bg_gt[0:1]);       labels.append('BG_GT')
                if pattern_gt is not None:
                    imgs.append(pattern_gt[0:1]);  labels.append('Pattern_GT')

                grid = vutils.make_grid(torch.cat(imgs, dim=0),
                                        nrow=len(imgs), normalize=True, padding=2)
                self.writer.add_image(f'samples/epoch_{epoch}_sample_{i}', grid, epoch)
                vutils.save_image(grid, os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i}.png'))

                for img_t, label in zip(imgs, labels):
                    self.writer.add_image(
                        f'components/{label}/epoch_{epoch}_sample_{i}',
                        vutils.make_grid(img_t, normalize=True, padding=2), epoch
                    )

        self.logger.info('Samples saved → %s', save_dir)


    def train(self):
        self.logger.info('Starting training | config: %s', self.config)
        self.create_dataloaders()

        start_epoch = 0
        if self.config.get('resume_from'):
            start_epoch = self.load_checkpoint(self.config['resume_from'])

        best_val_loss = float('inf')

        for epoch in range(start_epoch, self.epochs):
            train_losses = self.train_epoch(epoch)

            val_losses = {}
            if epoch % self.val_interval == 0:
                val_losses = self.validate(epoch)
                if epoch % (self.val_interval * 2) == 0:
                    self.save_sample_results(epoch)

            self.scheduler.step()
            self._log_epoch(epoch, train_losses, val_losses)

            is_best = False
            if val_losses and 'total' in val_losses:
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    is_best = True
                    self.logger.info('New best val loss: %.6f', best_val_loss)

            self.save_checkpoint(epoch, is_best)

        self.logger.info('Training complete.')
        self.writer.close()
        self.logger.info('TensorBoard: tensorboard --logdir %s', self.tensorboard_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='General Degradation Decomposition Training')

    # Data – folder paths
    parser.add_argument('--input_dir',       type=str, required=True,
                        help='Folder of degraded input images')
    parser.add_argument('--bg_dir',          type=str, default=None,
                        help='Folder of clean background GTs (optional)')
    parser.add_argument('--pattern_dir',     type=str, default=None,
                        help='Folder of degradation pattern GTs (optional)')
    parser.add_argument('--degradation_types', type=str, nargs='+', default=None,
                        help='List of degradation types (e.g., blur rain noise). '
                             'If not specified, will auto-detect from input_dir subfolders.')
    parser.add_argument('--val_input_dir',   type=str, default=None,
                        help='Validation input folder (optional)')
    parser.add_argument('--val_bg_dir',      type=str, default=None)
    parser.add_argument('--val_pattern_dir', type=str, default=None)
    parser.add_argument('--val_degradation_types', type=str, nargs='+', default=None,
                        help='List of validation degradation types (optional). '
                             'If not specified, will auto-detect from val_input_dir subfolders.')

    # Training
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=8)
    parser.add_argument('--val_batch_size', type=int,   default=4)
    parser.add_argument('--learning_rate',  type=float, default=1e-4)
    parser.add_argument('--weight_decay',   type=float, default=1e-5)
    parser.add_argument('--min_lr',         type=float, default=1e-7)
    parser.add_argument('--grad_clip',      type=float, default=1.0)

    # Loss weights
    parser.add_argument('--w_orthogonal', type=float, default=0.1)
    parser.add_argument('--w_pattern',    type=float, default=1.0)
    parser.add_argument('--w_bg',         type=float, default=1.0)
    parser.add_argument('--w_recon',      type=float, default=1.0)

    # Model
    parser.add_argument('--in_channels',   type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--image_height',  type=int, default=512)
    parser.add_argument('--image_width',   type=int, default=512)

    # Misc
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--save_interval',  type=int, default=20)
    parser.add_argument('--log_interval',   type=int, default=10)
    parser.add_argument('--val_interval',   type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/general_decomp')
    parser.add_argument('--tensorboard_dir',type=str, default=None)
    parser.add_argument('--device',         type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume_from',    type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = GeneralDecompositionTrainer(vars(args))
    trainer.train()
