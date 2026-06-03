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


class OrientationAwareBlock(nn.Module):
    def __init__(self, channels, num_orientations=4):
        super().__init__()
        self.num_orientations = num_orientations
        per_orient = channels // num_orientations
        self.per_orient = per_orient

        self.directional_convs = nn.ModuleList([
            self._make_orient_conv(per_orient, angle)
            for angle in range(num_orientations)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(per_orient * num_orientations, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.channel_adjust = None

    @staticmethod
    def _make_orient_conv(channels, angle_idx):
        kernel_sizes = [(1, 7), (1, 11), (1, 15), (1, 21)]
        k = kernel_sizes[angle_idx % len(kernel_sizes)]
        return nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=(k[0]//2, k[1]//2), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        C = x.shape[1]
        if C < self.num_orientations:
            return x
        usable = self.per_orient * self.num_orientations
        if C != usable:
            if self.channel_adjust is None or self.channel_adjust.in_channels != C:
                self.channel_adjust = nn.Conv2d(C, usable, 1, bias=False).to(x.device)
            x_in = self.channel_adjust(x)
        else:
            x_in = x

        parts = x_in.split(self.per_orient, dim=1)
        out_parts = [conv(part) for conv, part in zip(self.directional_convs, parts)]
        out = torch.cat(out_parts, dim=1)
        out = self.fusion(out)

        if out.shape[1] != C:
            return out
        return out + x


class ConvBottleneck(nn.Module):
    def __init__(self, channels, num_layers=4, dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 2][:num_layers]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dil = dilation_rates[i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dil, dilation=dil, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=dil, dilation=dil, bias=False),
                    nn.BatchNorm2d(channels),
                )
            )
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.out_proj(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=64, use_orient_block=False):
        super().__init__()
        bc = base_channels
        self.use_orient_block = use_orient_block
        self.enc1 = self._block(in_channels, bc,      dilation=1)
        self.enc2 = self._block(bc,          bc * 2,  dilation=2)
        self.enc3 = self._block(bc * 2,      bc * 4,  dilation=4)
        self.enc4 = self._block(bc * 4,      bc * 8,  dilation=2)
        self.pool = nn.MaxPool2d(2)

        if use_orient_block:
            self.orient2 = OrientationAwareBlock(bc * 2,  num_orientations=4)
            self.orient3 = OrientationAwareBlock(bc * 4,  num_orientations=4)
            self.orient4 = OrientationAwareBlock(bc * 8,  num_orientations=4)

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
        if self.use_orient_block:
            e2 = self.orient2(e2)
        e3 = self.enc3(self.pool(e2))
        if self.use_orient_block:
            e3 = self.orient3(e3)
        e4 = self.enc4(self.pool(e3))
        if self.use_orient_block:
            e4 = self.orient4(e4)
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
        g4 = self.up4(bottleneck)
        x = self.dec4(torch.cat([g4, e3], dim=1))

        g3 = self.up3(x)
        x = self.dec3(torch.cat([g3, e2], dim=1))

        g2 = self.up2(x)
        x = self.dec2(torch.cat([g2, e1], dim=1))
        return x


class FeatureDisentanglement(nn.Module):
    def __init__(self, in_channels, use_orient_block=False):
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
        self.use_orient_block = use_orient_block
        if use_orient_block:
            self.orient_pattern = OrientationAwareBlock(half, num_orientations=4)

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
        if self.use_orient_block:
            pat_feat = self.orient_pattern(pat_feat)
        orth_loss = self.orthogonal_loss(pat_feat, bg_feat)
        return pat_feat, bg_feat, orth_loss


class GeneralDecompositionNet(nn.Module):


    def __init__(self, in_channels: int = 3, base_channels: int = 64, bottleneck_type: str = "conv", use_orient_block: bool = False):
        super().__init__()
        bc = base_channels
        self.use_orient_block = use_orient_block

        self.encoder = UNetEncoder(in_channels, bc, use_orient_block=use_orient_block)
        if bottleneck_type == "conv":
            self.bottleneck = ConvBottleneck(bc * 8, num_layers=4)
        elif bottleneck_type == "transformer":
            self.bottleneck = TransformerBottleneck(bc * 8, num_heads=8, num_layers=4)
        else:
            raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")
        self.decoder = UNetDecoder(bc)

        self.disentangle = FeatureDisentanglement(bc, use_orient_block=use_orient_block)

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
##replace l2 with l1 loss, and add SSIM loss for better perceptual quality in rain degradation.

##realize ssim loss 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1.0):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    (_, channel, height, width) = img1.size()
    if window is None:
        window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) **2
    C2 = (0.03 * val_range)** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, val_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.val_range = val_range
        self.window = None

    def forward(self, x, y):
        if self.window is None or self.window.device != x.device:
            self.window = create_window(self.window_size, x.size(1)).to(x.device)
        return ssim(x, y, window=self.window, val_range=self.val_range)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred, target)

def frequency_high_freq_loss(pred, target, high_freq_ratio=0.3):
    """
    频域高频损失函数 - 增强高频细节恢复
    
    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        high_freq_ratio: 高频区域比例 (0-1), 控制高频提取的范围
    
    Returns:
        高频损失值
    """
    # 确保输入在合理范围内
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    
    batch_size, channels, height, width = pred.shape
    
    # 计算2D傅里叶变换
    def compute_dft(img):
        # 转换到频域
        dft = torch.fft.fft2(img, dim=(-2, -1))
        # 移到中心便于处理
        dft_shifted = torch.fft.fftshift(dft, dim=(-2, -1))
        return dft_shifted
    
    # 创建高频掩码
    def create_high_freq_mask(h, w, ratio):
        mask = torch.ones(h, w, device=pred.device)
        center_h, center_w = h // 2, w // 2
        
        # 计算要屏蔽的低频区域大小
        low_freq_size_h = int(h * ratio)
        low_freq_size_w = int(w * ratio)
        
        # 确保大小为奇数，以中心对称
        low_freq_size_h = low_freq_size_h if low_freq_size_h % 2 == 1 else low_freq_size_h + 1
        low_freq_size_w = low_freq_size_w if low_freq_size_w % 2 == 1 else low_freq_size_w + 1
        
        # 屏蔽中心低频区域
        start_h = center_h - low_freq_size_h // 2
        end_h = center_h + low_freq_size_h // 2 + 1
        start_w = center_w - low_freq_size_w // 2
        end_w = center_w + low_freq_size_w // 2 + 1
        
        mask[start_h:end_h, start_w:end_w] = 0
        return mask
    
    # 计算DFT
    pred_dft = compute_dft(pred)
    target_dft = compute_dft(target)
    
    # 创建高频掩码
    high_freq_mask = create_high_freq_mask(height, width, high_freq_ratio)
    high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 提取高频成分
    pred_high_freq = pred_dft * high_freq_mask
    target_high_freq = target_dft * high_freq_mask
    
    # 计算幅度谱损失
    pred_magnitude = torch.abs(pred_high_freq)
    target_magnitude = torch.abs(target_high_freq)
    
    # 使用L1损失计算高频差异
    freq_loss = F.l1_loss(pred_magnitude, target_magnitude)
    
    return freq_loss

def gradient_edge_loss(pred, target):
    """
    梯度边缘损失函数 - 保留图像边缘细节
    
    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
    
    Returns:
        边缘损失值
    """
    def compute_sobel_gradients(img):
        # Sobel算子
        sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], 
                              dtype=torch.float32, device=img.device)
        sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], 
                              dtype=torch.float32, device=img.device)
        
        # 对每个通道分别计算梯度
        grad_x_list = []
        grad_y_list = []
        
        for c in range(img.shape[1]):
            channel = img[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
        
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gradient_magnitude
    
    pred_grad = compute_sobel_gradients(pred)
    target_grad = compute_sobel_gradients(target)
    
    # 使用L1损失保持边缘
    edge_loss = F.l1_loss(pred_grad, target_grad)
    
    return edge_loss

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
        w_ssim:       float = 0.5,  # optional SSIM loss weight
        w_frequency:    float = 0.2,  # optional frequency-domain loss weight (not implemented here)
        w_edge:         float = 0.2,  # optional edge loss weight (not implemented here
    ):
        super().__init__()
        self.w_orthogonal = w_orthogonal
        self.w_pattern    = w_pattern
        self.w_bg         = w_bg
        self.w_recon      = w_recon
        self.w_ssim       = w_ssim
        self.ssim_loss = SSIMLoss()
        self.w_frequency = w_frequency
        self.w_edge = w_edge


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

        # # 2. Pattern L2
        # if pattern_gt is not None:
        #     losses['pattern_l2'] = F.mse_loss(pattern, pattern_gt)
        #     total = total + self.w_pattern * losses['pattern_l2']

        # 2. Pattern L1 + SSIM
        if pattern_gt is not None:
            losses['pattern_l1'] = F.l1_loss(pattern, pattern_gt)
            total = total + self.w_pattern * losses['pattern_l1']
            losses['pattern_ssim'] = self.ssim_loss(pattern, pattern_gt)
            total +=self.w_pattern * self.w_ssim * losses['pattern_ssim']
            losses['pattern_freq'] = frequency_high_freq_loss(pattern, pattern_gt,high_freq_ratio=0.3)
            total += self.w_pattern * self.w_frequency * losses['pattern_freq']
            losses['pattern_edge'] = gradient_edge_loss(pattern, pattern_gt)
            total += self.w_pattern * self.w_edge * losses['pattern_edge'] 

            
   
            


        # # 3. Background L2
        # if bg_gt is not None:
        #     losses['bg_l2'] = F.mse_loss(background, bg_gt)
        #     total = total + self.w_bg * losses['bg_l2']
        # 3. Background L1 + SSIM
        if bg_gt is not None:   
            losses['bg_l1'] = F.l1_loss(background, bg_gt)
            total = total + self.w_bg * losses['bg_l1']
            losses['bg_ssim'] = self.ssim_loss(background, bg_gt)
            total += self.w_bg * self.w_ssim * losses['bg_ssim']
            losses['bg_freq'] = frequency_high_freq_loss(background, bg_gt,high_freq_ratio=0.3)
            total += self.w_bg * self.w_frequency * losses['bg_freq']
            losses['bg_edge'] = gradient_edge_loss(background, bg_gt)
            total += self.w_bg * self.w_edge * losses['bg_edge']

        # # 4. Additive reconstruction L2:  pattern + background ≈ input
        # reconstruction = (pattern + background).clamp(0, 1)
        # losses['recon_l2'] = F.mse_loss(reconstruction, input_image)
        # total = total + self.w_recon * losses['recon_l2']

        # 4. Additive reconstruction L1 + SSIM
        reconstruction = (pattern + background).clamp(0, 1)
        losses['recon_l1'] = F.l1_loss(reconstruction, input_image)
        total = total + self.w_recon * losses['recon_l1']
        losses['recon_ssim'] = self.ssim_loss(reconstruction, input_image)
        total += self.w_recon * self.w_ssim * losses['recon_ssim']
        losses['recon_freq'] = frequency_high_freq_loss(reconstruction, input_image)
        total += self.w_recon * self.w_frequency * losses['recon_freq']
        losses['recon_edge'] = gradient_edge_loss(reconstruction, input_image)
        total += self.w_recon * self.w_edge * losses['recon_edge']

        losses['total'] = total
        return total, losses


def create_model(in_channels: int = 3, base_channels: int = 64, bottleneck_type: str = "conv", use_orient_block: bool = False) -> GeneralDecompositionNet:
    return GeneralDecompositionNet(in_channels=in_channels, base_channels=base_channels, bottleneck_type=bottleneck_type, use_orient_block=use_orient_block)


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
