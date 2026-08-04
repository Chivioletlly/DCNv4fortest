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
    ARCHITECTURE_VERSION,
    DIRECTION_BLOCK_TYPE,
    GeneralDecompositionNet,
    DecompositionLoss,
    count_parameters,
    validate_checkpoint_architecture,
)
from general_decomp.dataset import build_dataloader
from general_decomp.training_schedule import (
    format_patch_schedule,
    parse_patch_schedule,
    patch_size_for_epoch,
)


class GeneralDecompositionTrainer:

    def __init__(self, config: Dict):
        self.config = dict(config)
        self.config['architecture_version'] = ARCHITECTURE_VERSION
        self.config['direction_block_type'] = DIRECTION_BLOCK_TYPE
        self.patch_schedule = parse_patch_schedule(
            self.config.get('patch_schedule') or ()
        )
        self.config['patch_schedule'] = list(format_patch_schedule(self.patch_schedule))
        self.native_validation = bool(
            self.config.get('native_validation', False) or self.patch_schedule
        )
        self.config['native_validation'] = self.native_validation
        self.config['spatial_training_mode'] = (
            'progressive_patch' if self.patch_schedule else 'fixed_resize'
        )
        if self.patch_schedule:
            self.config['image_height'] = None
            self.config['image_width'] = None
        if self.config.get('resume_from') and self.config.get('init_from'):
            raise ValueError('resume_from and init_from are mutually exclusive')
        final_stage_after_training = (
            self.patch_schedule
            and self.patch_schedule[-1].start_epoch > self.config.get('epochs', 200)
        )
        if final_stage_after_training:
            raise ValueError('The final patch stage starts after the configured training epochs')
        if (
            self.native_validation
            and self.config.get('val_input_dir')
            and self.config.get('val_batch_size', 4) != 1
        ):
            raise ValueError('Native-resolution validation requires --val_batch_size 1')
        config = self.config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        if config.get('use_orient_block', False) and self.device.type != 'cuda':
            raise RuntimeError('DCNv4 training requires a Linux CUDA GPU')

        self.model = GeneralDecompositionNet(
            in_channels=config.get('in_channels', 3),
            base_channels=config.get('base_channels', 64),
            bottleneck_type=config.get('bottleneck_type', 'conv'),
            use_orient_block=config.get('use_orient_block', False),
        ).to(self.device)

        self.criterion = DecompositionLoss(
            w_orthogonal=config.get('w_orthogonal', 0.1),
            w_pattern=config.get('w_pattern', 1.0),
            w_bg=config.get('w_bg', 1.0),
            w_recon=config.get('w_recon', 1.0),
            w_ssim=config.get('w_ssim', 0.5),
            w_frequency=config.get('w_frequency', 0.2),
            w_edge=config.get('w_edge', 0.2),
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
        self.best_val_loss = float('inf')
        self.global_step = 0

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

    def _make_dataloader(
        self, input_dir, bg_dir, pattern_dir, degradation_types, sample_stems_dir,
        is_train
    ):
        bs = (
            self.config.get('batch_size', 8)
            if is_train else self.config.get('val_batch_size', 4)
        )
        patch_size = (
            patch_size_for_epoch(self.patch_schedule, 0)
            if is_train and self.patch_schedule else None
        )
        native_size = (is_train and self.patch_schedule) or (
            not is_train and self.native_validation
        )
        loader, ds = build_dataloader(
            input_dir=input_dir,
            bg_dir=bg_dir,
            pattern_dir=pattern_dir,
            degradation_types=degradation_types,
            sample_stems_dir=sample_stems_dir,
            height=None if native_size else self.config.get('image_height', 512),
            width=None if native_size else self.config.get('image_width', 512),
            patch_size=patch_size,
            batch_size=bs,
            augment=is_train,
            shuffle=is_train,
            num_workers=self.config.get('num_workers', 4),
        )
        return loader, ds

    def create_dataloaders(self):
        self.train_loader, self.train_dataset = self._make_dataloader(
            input_dir=self.config['input_dir'],
            bg_dir=self.config.get('bg_dir'),
            pattern_dir=self.config.get('pattern_dir'),
            degradation_types=self.config.get('degradation_types'),
            sample_stems_dir=self.config.get('train_stems_dir'),
            is_train=True,
        )
        self.logger.info('Train set: %d samples (degradation types: %s)',
                         len(self.train_dataset), self.train_dataset.degradation_types)

        self.val_loader = None
        if self.config.get('val_input_dir'):
            self.val_loader, val_ds = self._make_dataloader(
                input_dir=self.config['val_input_dir'],
                bg_dir=self.config.get('val_bg_dir'),
                pattern_dir=self.config.get('val_pattern_dir'),
                degradation_types=self.config.get('val_degradation_types'),
                sample_stems_dir=self.config.get('val_stems_dir'),
                is_train=False,
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
        raw_types = batch.get('degradation_type', 'unknown')
        degradation_types = GeneralDecompositionTrainer._normalize_degradation_types(
            raw_types, inp.shape[0]
        )
        return inp, bg_gt, pattern_gt, degradation_types

    @staticmethod
    def _normalize_degradation_types(raw_types, batch_size):
        """Return exactly one degradation label per sample in the batch."""
        if isinstance(raw_types, str):
            labels = [raw_types] * batch_size
        else:
            labels = [str(label) for label in raw_types]
        if len(labels) != batch_size:
            raise ValueError(
                'degradation_type count does not match batch size: '
                f'{len(labels)} != {batch_size}'
            )
        return [label.strip() or 'unknown' for label in labels]

    @staticmethod
    def _tensorboard_label(label):
        """Make a dataset label safe and stable as one TensorBoard path segment."""
        cleaned = ''.join(
            character if character.isalnum() or character in {'-', '_', '.'} else '_'
            for character in str(label).strip()
        )
        return cleaned or 'unknown'

    @staticmethod
    def _add_stat(accumulator, key, value_sum, sample_count):
        current_sum, current_count = accumulator.get(key, (0.0, 0))
        accumulator[key] = (
            current_sum + float(value_sum),
            current_count + int(sample_count),
        )

    @classmethod
    def _accumulate_loss_stats(
        cls, accumulator, loss_dict, per_sample_losses, degradation_types
    ):
        """Accumulate sample-weighted overall and exact per-type image losses."""
        batch_size = len(degradation_types)
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                cls._add_stat(
                    accumulator,
                    f'overall/{name}',
                    value.detach().item() * batch_size,
                    batch_size,
                )

        grouped_indices = {}
        for index, label in enumerate(degradation_types):
            safe_label = cls._tensorboard_label(label)
            grouped_indices.setdefault(safe_label, []).append(index)

        for label, indices in grouped_indices.items():
            for name, values in per_sample_losses.items():
                cls._add_stat(
                    accumulator,
                    f'by_type/{label}/{name}',
                    values[indices].sum().item(),
                    len(indices),
                )

    @staticmethod
    def _average_stats(accumulator):
        return {
            key: value_sum / sample_count
            for key, (value_sum, sample_count) in accumulator.items()
            if sample_count > 0
        }

    @staticmethod
    def _losses_to_cpu(per_sample_losses):
        """Copy all per-sample metrics with one small device synchronization."""
        loss_names = tuple(per_sample_losses)
        loss_matrix = torch.stack(
            [per_sample_losses[name] for name in loss_names], dim=1
        ).detach().cpu()
        return {
            name: loss_matrix[:, index] for index, name in enumerate(loss_names)
        }

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def _set_training_patch(self, epoch: int):
        if not self.patch_schedule:
            return None
        patch_size = patch_size_for_epoch(self.patch_schedule, epoch)
        if self.train_dataset.patch_size != patch_size:
            self.train_dataset.set_patch_size(patch_size)
            self.logger.info(
                'Progressive patch stage: epoch %d uses %dx%d crops',
                epoch + 1,
                patch_size,
                patch_size,
            )
        self.writer.add_scalar('train_epoch/patch_size', patch_size, epoch + 1)
        return patch_size

    def _forward_and_loss(self, inp, bg_gt, pattern_gt):
        pattern, background, orth_loss = self.model(inp)
        total_loss, loss_dict, per_sample_losses = self.criterion(
            pattern,
            background,
            orth_loss,
            inp,
            pattern_gt,
            bg_gt,
            return_per_sample=True,
        )
        per_sample_losses = {
            name: values.detach() for name, values in per_sample_losses.items()
        }
        return pattern, background, total_loss, loss_dict, per_sample_losses

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        epoch_stats: Dict = {}
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                inp, bg_gt, pattern_gt, degradation_types = self._unpack_batch(
                    batch, self.device
                )

                self.optimizer.zero_grad()
                pattern, background, total_loss, loss_dict, per_sample_losses = (
                    self._forward_and_loss(inp, bg_gt, pattern_gt)
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get('grad_clip', 1.0)
                )
                self.optimizer.step()
                per_sample_losses = self._losses_to_cpu(per_sample_losses)

                self._accumulate_loss_stats(
                    epoch_stats, loss_dict, per_sample_losses, degradation_types
                )

                pbar.set_postfix({
                    'loss':  f'{total_loss.item():.4f}',
                    'recon': f'{loss_dict.get("recon_l1", 0):.4f}',
                    'orth':  f'{loss_dict.get("orthogonal", 0):.4f}',
                    'lr':    f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                })

                if batch_idx % self.log_interval == 0:
                    self._log_batch(
                        epoch,
                        batch_idx,
                        loss_dict,
                        per_sample_losses,
                        num_batches,
                        degradation_types,
                    )
                self.global_step += 1

        return self._average_stats(epoch_stats)

    def validate(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_stats: Dict = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                inp, bg_gt, pattern_gt, degradation_types = self._unpack_batch(
                    batch, self.device
                )
                pattern, background, _, loss_dict, per_sample_losses = (
                    self._forward_and_loss(inp, bg_gt, pattern_gt)
                )
                per_sample_losses = self._losses_to_cpu(per_sample_losses)

                recon = (pattern + background).clamp(0, 1)
                recon_mse = (recon - inp).square().flatten(1).mean(dim=1)
                loss_dict = dict(loss_dict)
                loss_dict['recon_mse'] = recon_mse.mean()
                per_sample_losses = dict(per_sample_losses)
                per_sample_losses['recon_mse'] = recon_mse.cpu()
                if bg_gt is not None:
                    bg_mse = (background - bg_gt).square().flatten(1).mean(dim=1)
                    bg_psnr = -10.0 * torch.log10(bg_mse.clamp_min(1e-10))
                    bg_ssim = 1.0 - self.criterion.ssim_loss(
                        background, bg_gt, reduction='none'
                    )
                    loss_dict['bg_psnr'] = bg_psnr.mean()
                    loss_dict['bg_ssim_score'] = bg_ssim.mean()
                    per_sample_losses['bg_psnr'] = bg_psnr.cpu()
                    per_sample_losses['bg_ssim_score'] = bg_ssim.cpu()
                self._accumulate_loss_stats(
                    val_stats, loss_dict, per_sample_losses, degradation_types
                )

        return self._average_stats(val_stats)


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'current_patch_size': self.train_dataset.patch_size,
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
        validate_checkpoint_architecture(ckpt.get('config', {}))
        checkpoint_schedule = parse_patch_schedule(
            ckpt.get('config', {}).get('patch_schedule') or ()
        )
        if checkpoint_schedule != self.patch_schedule:
            raise ValueError(
                'Checkpoint patch_schedule does not match the current run. '
                'Use --init_from to load model weights with a new schedule.'
            )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get(
            'global_step', (epoch + 1) * len(self.train_loader)
        )
        self.logger.info('Resumed after epoch %d (%s)', epoch + 1, path)
        return epoch + 1

    def load_initial_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Initialization checkpoint not found: {path}')
        ckpt = torch.load(path, map_location=self.device)
        config = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
        validate_checkpoint_architecture(config)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state_dict)
        self.logger.info(
            'Loaded model initialization from %s; optimizer and epoch were reset', path
        )



    def _log_batch(
        self,
        epoch,
        batch_idx,
        loss_dict,
        per_sample_losses,
        num_batches,
        degradation_types,
    ):
        step = self.global_step
        # 总体损失（不区分退化类型）
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.writer.add_scalar(f'train_batch/overall/{k}', v.item(), step)
        self.writer.add_scalar(
            'train_batch/lr', self.optimizer.param_groups[0]['lr'], step
        )

        # 按退化类型记录
        grouped_indices = {}
        for index, label in enumerate(degradation_types):
            safe_label = self._tensorboard_label(label)
            grouped_indices.setdefault(safe_label, []).append(index)
        for label, indices in grouped_indices.items():
            for name, values in per_sample_losses.items():
                self.writer.add_scalar(
                    f'train_batch/by_type/{label}/{name}',
                    values[indices].mean().item(),
                    step,
                )

        loss_str = ' | '.join(
            f'{k}: {v.item():.4f}' for k, v in loss_dict.items() if isinstance(v, torch.Tensor)
        )
        labels = ','.join(sorted(set(degradation_types)))
        self.logger.info(
            'Epoch %3d | Batch %4d/%4d | types=%s | %s',
            epoch + 1,
            batch_idx + 1,
            num_batches,
            labels,
            loss_str,
        )

    def _log_epoch(self, epoch, train_losses, val_losses):
        # 总体损失（不区分退化类型）
        overall_train = {
            key[len('overall/'):]: value
            for key, value in train_losses.items()
            if key.startswith('overall/')
        }
        overall_val = {
            key[len('overall/'):]: value
            for key, value in val_losses.items()
            if key.startswith('overall/')
        }

        self.logger.info('Epoch %3d Train | %s', epoch + 1,
                         ' | '.join(f'{k}: {v:.4f}' for k, v in overall_train.items()))
        if overall_val:
            self.logger.info('Epoch %3d Val   | %s', epoch + 1,
                             ' | '.join(f'{k}: {v:.4f}' for k, v in overall_val.items()))

        # 记录总体损失
        for key, value in train_losses.items():
            self.writer.add_scalar(f'train_epoch/{key}', value, epoch + 1)
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val_epoch/{key}', value, epoch + 1)

        # 按退化类型记录
        self.writer.add_scalar(
            'train_epoch/learning_rate',
            self.optimizer.param_groups[0]['lr'],
            epoch + 1,
        )

        self.writer.flush()

    def save_sample_results(self, epoch: int, num_samples: int = 4):
        if self.val_loader is None:
            return

        self.model.eval()
        save_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(save_dir, exist_ok=True)

        saved_samples = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if saved_samples >= num_samples:
                    break
                inp, bg_gt, pattern_gt, degradation_types = self._unpack_batch(
                    batch, self.device
                )
                pattern, background, _ = self.model(inp)
                reconstructed = (pattern + background).clamp(0, 1)

                for sample_index in range(inp.shape[0]):
                    if saved_samples >= num_samples:
                        break
                    images = [
                        inp[sample_index:sample_index + 1],
                        pattern[sample_index:sample_index + 1],
                        background[sample_index:sample_index + 1],
                        reconstructed[sample_index:sample_index + 1],
                    ]
                    labels = ['input', 'pattern', 'background', 'reconstructed']

                    if bg_gt is not None:
                        images.append(bg_gt[sample_index:sample_index + 1])
                        labels.append('background_gt')
                    if pattern_gt is not None:
                        images.append(pattern_gt[sample_index:sample_index + 1])
                        labels.append('pattern_gt')

                    tag = f'val_samples/sample_{saved_samples:02d}'
                    grid = vutils.make_grid(
                        torch.cat(images, dim=0),
                        nrow=len(images),
                        normalize=False,
                        padding=2,
                    )
                    self.writer.add_image(f'{tag}/comparison', grid, epoch + 1)
                    self.writer.add_text(
                        f'{tag}/degradation_type',
                        degradation_types[sample_index],
                        epoch + 1,
                    )
                    vutils.save_image(
                        grid,
                        os.path.join(
                            save_dir,
                            f'epoch_{epoch + 1:03d}_sample_{saved_samples:02d}.png',
                        ),
                    )

                    for image, label in zip(images, labels):
                        self.writer.add_image(
                            f'{tag}/{label}', image[0].clamp(0, 1), epoch + 1
                        )
                    saved_samples += 1

        self.logger.info('Samples saved → %s', save_dir)


    def train(self):
        self.logger.info('Starting training | config: %s', self.config)
        self.create_dataloaders()

        start_epoch = 0
        if self.config.get('resume_from'):
            start_epoch = self.load_checkpoint(self.config['resume_from'])
        elif self.config.get('init_from'):
            self.load_initial_weights(self.config['init_from'])

        for epoch in range(start_epoch, self.epochs):
            self._set_training_patch(epoch)
            train_losses = self.train_epoch(epoch)

            val_losses = {}
            if epoch % self.val_interval == 0:
                val_losses = self.validate(epoch)
                if epoch % (self.val_interval * 2) == 0:
                    self.save_sample_results(epoch)

            self.scheduler.step()
            self._log_epoch(epoch, train_losses, val_losses)

            is_best = False
            if 'overall/total' in val_losses:
                if val_losses['overall/total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['overall/total']
                    is_best = True
                    self.logger.info('New best val loss: %.6f', self.best_val_loss)

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
    parser.add_argument(
        '--train_stems_dir', type=str, default=None,
        help='Optional image directory whose filename stems select the training split',
    )
    parser.add_argument('--val_input_dir',   type=str, default=None,
                        help='Validation input folder (optional)')
    parser.add_argument('--val_bg_dir',      type=str, default=None)
    parser.add_argument('--val_pattern_dir', type=str, default=None)
    parser.add_argument('--val_degradation_types', type=str, nargs='+', default=None,
                        help='List of validation degradation types (optional). '
                             'If not specified, will auto-detect from val_input_dir subfolders.')
    parser.add_argument(
        '--val_stems_dir', type=str, default=None,
        help='Optional image directory whose filename stems select the validation split',
    )

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
    parser.add_argument('--w_ssim',       type=float, default=0.5)
    parser.add_argument('--w_frequency',   type=float, default=0.2)
    parser.add_argument('--w_edge',        type=float, default=0.2)
    # Model
    parser.add_argument('--in_channels',   type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--bottleneck_type', type=str, default='conv',
                        choices=['conv', 'transformer'],
                        help='Bottleneck type: conv (fast) or transformer (accurate)')
    parser.add_argument('--use_orient_block', action='store_true',
                        help='Use residual DCNv4 blocks in encoder and pattern branch')
    parser.add_argument('--image_height',  type=int, default=512)
    parser.add_argument('--image_width',   type=int, default=512)
    parser.add_argument(
        '--patch_schedule',
        type=str,
        nargs='+',
        default=None,
        metavar='EPOCH:SIZE',
        help=(
            'Enable Restormer-style progressive crops from original images. '
            'Example: 1:128 11:160 26:192 41:256 61:320 81:384. '
            'Patch sizes must be multiples of 8.'
        ),
    )
    parser.add_argument(
        '--native_validation',
        action='store_true',
        help='Validate at each image original size (requires --val_batch_size 1)',
    )

    # Misc
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--save_interval',  type=int, default=20)
    parser.add_argument('--log_interval',   type=int, default=10)
    parser.add_argument('--val_interval',   type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/general_decomp')
    parser.add_argument('--tensorboard_dir',type=str, default=None)
    parser.add_argument('--device',         type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        '--resume_from', type=str, default=None,
        help='Resume model, optimizer, scheduler, epoch, and the same patch schedule',
    )
    checkpoint_group.add_argument(
        '--init_from', type=str, default=None,
        help='Load compatible model weights only and start a new training run',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = GeneralDecompositionTrainer(vars(args))
    trainer.train()
