import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=8, num_layers=4, max_height=128, max_width=128):
        super().__init__()
        self.channels = channels
        self.max_height = max_height
        self.max_width = max_width

        self.to_patch = nn.Conv2d(channels, channels, 1)
        self.layers = nn.ModuleList(
            [TransformerBlock(channels, num_heads) for _ in range(num_layers)]
        )
        self.pos_h = nn.Parameter(torch.randn(1, max_height, 1, channels))
        self.pos_w = nn.Parameter(torch.randn(1, 1, max_width, channels))
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        if H > self.max_height or W > self.max_width:
            raise ValueError(
                f"Feature map {H}x{W} exceeds max supported size "
                f"{self.max_height}x{self.max_width}"
            )
        x = self.to_patch(x)
        seq = rearrange(x, 'b c h w -> b (h w) c')
        pos = self.pos_h[:, :H] + self.pos_w[:, :, :W]      # [1, H, W, C]
        pos = rearrange(pos, '1 h w c -> 1 (h w) c')
        seq = seq + pos
        for layer in self.layers:
            seq = layer(seq)
        x = rearrange(seq, 'b (h w) c -> b c h w', h=H, w=W)
        return self.out_proj(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        bc = base_channels
        self.enc1 = self._block(in_channels, bc,      dilation=1)
        self.enc2 = self._block(bc,          bc * 2,  dilation=2)
        self.enc3 = self._block(bc * 2,      bc * 4,  dilation=4)
        self.enc4 = self._block(bc * 4,      bc * 8,  dilation=2)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(in_ch, out_ch, dilation):
        p = dilation
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=p, dilation=dilation),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=p, dilation=dilation),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return [e1, e2, e3, e4]


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        bc = base_channels
        self.up4 = nn.ConvTranspose2d(bc * 8, bc * 4, 4, stride=2, padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 2, 4, stride=2, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(bc * 2, bc,     4, stride=2, padding=1, bias=False)
        self.dec4 = self._block(bc * 8, bc * 4)
        self.dec3 = self._block(bc * 4, bc * 2)
        self.dec2 = self._block(bc * 2, bc)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, bottleneck, enc_feats):
        e1, e2, e3 = enc_feats
        x = self.dec4(torch.cat([self.up4(bottleneck), e3], dim=1))
        x = self.dec3(torch.cat([self.up3(x),          e2], dim=1))
        x = self.dec2(torch.cat([self.up2(x),          e1], dim=1))
        return x


class FeatureDisentanglement(nn.Module):
    """Split decoded features into two orthogonal branches."""

    def __init__(self, in_channels):
        super().__init__()
        half = in_channels // 2
        self.pattern_branch = nn.Sequential(
            nn.Conv2d(in_channels, half, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(half,        half, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.bg_branch = nn.Sequential(
            nn.Conv2d(in_channels, half, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(half,        half, 3, padding=1), nn.ReLU(inplace=True),
        )

    @staticmethod
    def orthogonal_loss(a, b):
        """Cross-correlation orthogonality loss between two feature maps."""
        a_flat = a.view(a.size(0), a.size(1), -1)          # [B, C, HW]
        b_flat = b.view(b.size(0), b.size(1), -1)
        a_c = a_flat - a_flat.mean(dim=2, keepdim=True)
        b_c = b_flat - b_flat.mean(dim=2, keepdim=True)
        cross = torch.bmm(a_c, b_c.transpose(1, 2))        # [B, C, C]
        return cross.abs().mean()

    def forward(self, x):
        pat_feat = self.pattern_branch(x)
        bg_feat  = self.bg_branch(x)
        orth_loss = self.orthogonal_loss(pat_feat, bg_feat)
        return pat_feat, bg_feat, orth_loss


class GeneralDecompositionNet(nn.Module):
    

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        bc = base_channels

        self.encoder   = UNetEncoder(in_channels, bc)
        self.bottleneck = TransformerBottleneck(bc * 8, num_heads=8, num_layers=4)
        self.decoder   = UNetDecoder(bc)

        self.disentangle = FeatureDisentanglement(bc)

        half = bc // 2

        # Fusion before output heads
        self.fusion = nn.Sequential(
            nn.Conv2d(bc, half, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(half, bc, 3, padding=1),
        )

        # Degradation pattern head: no explicit bound (can be any non-negative value)
        self.pattern_head = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(half, in_channels, 1),
            nn.ReLU(),           # non-negative pattern
        )

        # Background head: clean image lives in [0, 1]
        self.bg_head = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(half, in_channels, 1),
            nn.Sigmoid(),        # constrained to [0, 1]
        )

    def forward(self, x):

        enc_feats = self.encoder(x)

        bottleneck = self.bottleneck(enc_feats[-1])
        dec_feat   = self.decoder(bottleneck, enc_feats[:-1])

        pat_feat, bg_feat, orth_loss = self.disentangle(dec_feat)

        fused = self.fusion(torch.cat([pat_feat, bg_feat], dim=1))
        pat_refined, bg_refined = fused.chunk(2, dim=1)

        pattern    = self.pattern_head(pat_refined)
        background = self.bg_head(bg_refined)

        return pattern, background, orth_loss


class DecompositionLoss(nn.Module):
    """
    Simplified decomposition loss with four terms:

    1. orthogonal   – feature cross-correlation penalty (encourages independence)
    2. pattern_l2   – supervised L2 on the degradation pattern
    3. bg_l2        – supervised L2 on the background
    4. recon_l2     – L2 between (pattern + background) and the input image

    All supervised terms are optional; when GT is not provided they are skipped.
    """

    def __init__(
        self,
        w_orthogonal: float = 0.1,
        w_pattern:    float = 1.0,
        w_bg:         float = 1.0,
        w_recon:      float = 1.0,
    ):
        super().__init__()
        self.w_orthogonal = w_orthogonal
        self.w_pattern    = w_pattern
        self.w_bg         = w_bg
        self.w_recon      = w_recon

    def forward(
        self,
        pattern,
        background,
        orth_loss,
        input_image,
        pattern_gt=None,
        bg_gt=None,
    ):
       
        losses = {}
        total  = 0.0

        # 1. Orthogonal loss
        losses['orthogonal'] = orth_loss
        total = total + self.w_orthogonal * orth_loss

        # 2. Pattern L2
        if pattern_gt is not None:
            losses['pattern_l2'] = F.mse_loss(pattern, pattern_gt)
            total = total + self.w_pattern * losses['pattern_l2']

        # 3. Background L2
        if bg_gt is not None:
            losses['bg_l2'] = F.mse_loss(background, bg_gt)
            total = total + self.w_bg * losses['bg_l2']

        # 4. Additive reconstruction L2:  pattern + background ≈ input
        reconstruction = (pattern + background).clamp(0, 1)
        losses['recon_l2'] = F.mse_loss(reconstruction, input_image)
        total = total + self.w_recon * losses['recon_l2']

        losses['total'] = total
        return total, losses


def create_model(in_channels: int = 3, base_channels: int = 64) -> GeneralDecompositionNet:
    return GeneralDecompositionNet(in_channels=in_channels, base_channels=base_channels)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'size_mb': total * 4 / (1024 ** 2)}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model().to(device)
    print('Parameters:', count_parameters(model))

    B, C, H, W = 2, 3, 256, 256
    x          = torch.rand(B, C, H, W, device=device)
    pattern_gt = torch.rand(B, C, H, W, device=device) * 0.3   # sparse pattern
    bg_gt      = torch.rand(B, C, H, W, device=device)

    model.eval()
    with torch.no_grad():
        pattern, background, orth_loss = model(x)

    print(f'pattern    shape: {pattern.shape},    range [{pattern.min():.3f}, {pattern.max():.3f}]')
    print(f'background shape: {background.shape}, range [{background.min():.3f}, {background.max():.3f}]')
    print(f'orth_loss: {orth_loss.item():.6f}')

    criterion = DecompositionLoss()
    total, loss_dict = criterion(pattern, background, orth_loss, x, pattern_gt, bg_gt)
    print('\nLoss breakdown:')
    for k, v in loss_dict.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f'  {k}: {val:.6f}')
