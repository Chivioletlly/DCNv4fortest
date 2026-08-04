from pathlib import Path

import pytest
import torch
import torchvision.transforms.functional as TF

from dataset import GeneralDecompDataset, build_dataloader
from general_decomposition_model import FeatureDisentanglement, GeneralDecompositionNet
from image_geometry import crop_to_size, pad_to_multiple
from training_schedule import parse_patch_schedule, patch_size_for_epoch


def _save_coordinate_image(path: Path, height: int, width: int):
    values = torch.arange(height * width, dtype=torch.int64).reshape(height, width)
    image = ((values % 251) / 250.0).unsqueeze(0).repeat(3, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(image).save(path)


def test_pad_and_crop_round_trip_for_rain13k_shape():
    image = torch.rand(1, 3, 321, 481)
    padded, size = pad_to_multiple(image)

    assert padded.shape[-2:] == (328, 488)
    assert torch.equal(crop_to_size(padded, size), image)


def test_model_accepts_non_multiple_native_size():
    model = GeneralDecompositionNet(base_channels=16).eval()
    image = torch.rand(1, 3, 31, 45)

    with torch.no_grad():
        pattern, background, orthogonal_loss = model(image)

    assert pattern.shape == image.shape
    assert background.shape == image.shape
    assert orthogonal_loss.ndim == 0


def test_orthogonal_loss_is_normalized_for_spatial_size():
    pattern = torch.rand(1, 4, 2, 3)
    background = torch.rand(1, 4, 2, 3)
    enlarged_pattern = pattern.repeat_interleave(2, -2).repeat_interleave(2, -1)
    enlarged_background = background.repeat_interleave(2, -2).repeat_interleave(2, -1)

    small_loss = FeatureDisentanglement.orthogonal_loss(pattern, background)
    enlarged_loss = FeatureDisentanglement.orthogonal_loss(
        enlarged_pattern, enlarged_background
    )

    assert torch.allclose(small_loss, enlarged_loss, atol=1e-6)


def test_progressive_patch_crop_stays_aligned(tmp_path):
    input_path = tmp_path / 'inputs' / 'input' / 'sample.png'
    target_path = tmp_path / 'targets' / 'sample.png'
    _save_coordinate_image(input_path, 23, 29)
    _save_coordinate_image(target_path, 23, 29)

    dataset = GeneralDecompDataset(
        str(tmp_path / 'inputs'),
        bg_dir=str(tmp_path / 'targets'),
        degradation_types=['input'],
        height=None,
        width=None,
        patch_size=32,
        augment=True,
    )
    sample = dataset[0]

    assert sample['input_image'].shape == (3, 32, 32)
    assert torch.equal(sample['input_image'], sample['lol_gt'])


def test_flat_directory_keeps_native_size(tmp_path):
    _save_coordinate_image(tmp_path / 'flat' / 'sample.png', 31, 45)
    dataset = GeneralDecompDataset(
        str(tmp_path / 'flat'), height=None, width=None
    )

    sample = dataset[0]

    assert dataset.degradation_types == ['input']
    assert sample['input_image'].shape == (3, 31, 45)


def test_split_selector_uses_stems_but_loads_original_images(tmp_path):
    _save_coordinate_image(tmp_path / 'native' / 'input' / 'train.png', 31, 45)
    _save_coordinate_image(tmp_path / 'native' / 'input' / 'held_out.png', 33, 47)
    _save_coordinate_image(tmp_path / 'split' / 'train.png', 8, 8)
    dataset = GeneralDecompDataset(
        str(tmp_path / 'native'),
        degradation_types=['input'],
        sample_stems_dir=str(tmp_path / 'split'),
        height=None,
        width=None,
    )

    assert len(dataset) == 1
    assert dataset[0]['filename'] == 'train'
    assert dataset[0]['input_image'].shape == (3, 31, 45)


def test_mixed_native_sizes_report_batch_size_solution(tmp_path):
    _save_coordinate_image(tmp_path / 'flat' / 'a.png', 24, 32)
    _save_coordinate_image(tmp_path / 'flat' / 'b.png', 25, 32)
    loader, _ = build_dataloader(
        str(tmp_path / 'flat'),
        height=None,
        width=None,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    with pytest.raises(ValueError, match='batch_size=1'):
        next(iter(loader))


def test_progressive_patch_schedule_lookup():
    schedule = parse_patch_schedule(
        ['1:128', '11:160', '26:192', '41:256', '61:320', '81:384']
    )

    assert patch_size_for_epoch(schedule, 0) == 128
    assert patch_size_for_epoch(schedule, 9) == 128
    assert patch_size_for_epoch(schedule, 10) == 160
    assert patch_size_for_epoch(schedule, 99) == 384


@pytest.mark.parametrize(
    'entries',
    (
        ['2:128'],
        ['1:130'],
        ['1:192', '10:128'],
        ['1:128', '1:160'],
        ['bad'],
    ),
)
def test_invalid_patch_schedules_are_rejected(entries):
    with pytest.raises(ValueError):
        parse_patch_schedule(entries)
