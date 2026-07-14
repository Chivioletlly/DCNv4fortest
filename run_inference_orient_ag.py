import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
from general_decomposition_model import (
    GeneralDecompositionNet,
    validate_checkpoint_architecture,
)

ckpt_path = os.path.join(os.path.dirname(__file__), 'conv_rain_512_orient_ag', 'latest.pth')
input_dir = os.path.join(os.path.dirname(__file__), 'test_rain_structured', 'rain')
save_dir = os.path.join(os.path.dirname(__file__), 'result_orient_ag')
os.makedirs(save_dir, exist_ok=True)

print(f'Loading checkpoint: {ckpt_path}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
config = ckpt['config']
validate_checkpoint_architecture(config)
if config.get('use_orient_block', False) and device.type != 'cuda':
    raise RuntimeError('DCNv4 inference requires a CUDA GPU')
print(f'Config: use_orient_block={config.get("use_orient_block", False)}, '
      f'direction_block_type={config.get("direction_block_type", "none")}')

model = GeneralDecompositionNet(
    in_channels=config.get('in_channels', 3),
    base_channels=config.get('base_channels', 64),
    bottleneck_type='conv',
    use_orient_block=config.get('use_orient_block', False),
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Model loaded successfully on {device}')

h = config.get('image_height', 512)
w = config.get('image_width', 512)
tf = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])

files = sorted(
    f
    for f in os.listdir(input_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
)
print(f'Found {len(files)} images in {input_dir}')

for i, fname in enumerate(files):
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('RGB')
    inp = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pattern, background, _ = model(inp)

    stem = os.path.splitext(fname)[0]
    vutils.save_image(inp.cpu(), os.path.join(save_dir, f'{stem}_degraded.png'))
    vutils.save_image(background.cpu(), os.path.join(save_dir, f'{stem}_derained.png'))
    print(f'[{i+1}/{len(files)}] {fname} -> {stem}_derained.png')

print(f'All done! Results saved to {save_dir}')
