import torch
from torch import nn
from torch.nn import functional as F


class LearnedUpsampleBlock(nn.Module):
    '''Learned upsampler that fuses a denoised LR image with the noisy HR
    center frame via a residual connection.

    Architecture:
        1. PixelShuffle upsample of the LR denoised image  (3 → feat → 3·sf²)
        2. Concatenate with the noisy HR center frame       (3 + 3 = 6 ch)
        3. Fusion conv layers to combine clean LR structure
           with HR high-frequency detail                    (6 → feat → 3)

    The HR residual gives the network direct access to edges and texture
    from the original resolution; the LR branch supplies the clean
    low-frequency content.  The fusion layers learn to keep HR detail
    while suppressing noise.

    Args:
        scale_factor: spatial upsampling factor (must match downscale_factor)
        feat_ch: number of intermediate feature channels
    '''

    def __init__(self, scale_factor: int = 2, feat_ch: int = 32):
        super(LearnedUpsampleBlock, self).__init__()
        self.scale_factor = scale_factor

        # --- Learned upsample path (LR → HR) ---
        # Conv to expand channels, then PixelShuffle to increase spatial res
        self.upsample = nn.Sequential(
            nn.Conv2d(3, feat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 3 * scale_factor * scale_factor,
                      kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(scale_factor),
        )

        # --- Fusion path (upsampled LR + noisy HR center → clean HR) ---
        # Input: 3 (upsampled denoised) + 3 (noisy HR center) = 6 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(6, feat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 3, kernel_size=3, padding=1, bias=False),
        )

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, denoised_lr: torch.Tensor,
                noisy_hr_center: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            denoised_lr:      [N, 3, H_lr, W_lr] — clean, low-resolution
            noisy_hr_center:  [N, 3, H, W]        — noisy center frame at
                              original resolution (provides HR detail)
        Returns:
            clean_hr:         [N, 3, H, W]
        '''
        # Learned upsample: LR → HR spatial size
        up = self.upsample(denoised_lr)  # [N, 3, H_up, W_up]

        # Handle slight size mismatches from align_to_multiple rounding
        _, _, H_hr, W_hr = noisy_hr_center.shape
        if up.shape[-2:] != noisy_hr_center.shape[-2:]:
            up = F.interpolate(up, size=(H_hr, W_hr), mode='bilinear',
                               align_corners=False)

        # Fuse upsampled clean LR with noisy HR detail
        fused = torch.cat([up, noisy_hr_center], dim=1)  # [N, 6, H, W]
        residual = self.fusion(fused)                     # [N, 3, H, W]

        # Residual learning: start from the noisy center frame,
        # predict a correction
        clean_hr = noisy_hr_center - residual

        return clean_hr.clamp(0., 1.)
