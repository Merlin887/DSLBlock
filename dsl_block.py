import torch
from torch import nn
from torch.nn import functional as F

from up_block import UpBlock
from output_cv_block import OutputCvBlock
from down_block import DownBlock
from input_cv_block import InputCvBlock
from learned_upsample_block import LearnedUpsampleBlock


class DSLBlock(nn.Module):
    """DSLBlock with built-in downscale → UNet → learned upsample.

    Same interface as DenBlock (drop-in replacement) but processes at
    reduced resolution internally for speed, with a LearnedUpsampleBlock
    to reconstruct HR output using the noisy HR center frame as guide.

    Pipeline:
        1. Save noisy HR center frame for the learned upsampler's residual path
        2. Bicubic-downsample input frames + noise_map to LR
        3. InputCvBlock → DownBlock×2 → UpBlock×2 → OutputCvBlock  (all at LR)
           → produces denoised LR  [N, 3, H_lr, W_lr]
        4. LearnedUpsampleBlock(denoised_lr, noisy_hr_center) → clean HR

    Args:
        channels:         encoder/decoder channel widths [c0, c1, c2]
        interm_ch:        intermediate channels in InputCvBlock
        simp_cv:          use LiteCvBlock (single conv) vs CvBlock (double)
        num_input_frames: number of input frames (3 for temp1/temp2)
        use_noise_map:    use noise map in denoising process
        use_depthwise:    depthwise-separable first conv in InputCvBlock
        downscale_factor: spatial downscale ratio (default 2)
        upsample_feat_ch: feature channels in the LearnedUpsampleBlock
    """

    def __init__(self, channels, interm_ch, simp_cv, num_input_frames=3,
                 use_noise_map=True, use_depthwise=True,
                 downscale_factor=2, upsample_feat_ch=32):
        super(DSLBlock, self).__init__()
        self.chs_lyr0 = channels[0]
        self.chs_lyr1 = channels[1]
        self.chs_lyr2 = channels[2]
        self.use_noise_map = use_noise_map
        self.downscale_factor = downscale_factor

        # --- UNet (runs at LR) ---
        self.inc = InputCvBlock(num_in_frames=num_input_frames,
                                interm_ch=interm_ch,
                                out_ch=self.chs_lyr0,
                                use_noise_map=use_noise_map,
                                use_depthwise=use_depthwise)

        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, simplifed_cv=simp_cv)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, simplifed_cv=simp_cv)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, simplifed_cv=simp_cv)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, simplifed_cv=simp_cv)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        # --- Learned upsampler (LR denoised → HR, with HR residual) ---
        self.learned_upsample = LearnedUpsampleBlock(
            scale_factor=downscale_factor,
            feat_ch=upsample_feat_ch,
        )

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    @staticmethod
    def align_to_multiple(value: int, multiple: int) -> int:
        '''Round *up* to the nearest multiple (ensures UNet compatibility).'''
        return ((value + multiple - 1) // multiple) * multiple

    def forward(self, input_tensors: list, noise_map):
        '''Args:
            input_tensors: list of Tensors [N, C, H, W] in [0., 1.] (HR)
            noise_map: Tensor [N, 1, H, W] in [0., 1.] (HR)
        Returns:
            clean HR frame [N, 3, H, W]
        '''
        # --- 1. Save HR center frame for learned upsampler residual ---
        hr_center = input_tensors[len(input_tensors) // 2]
        _, _, H, W = hr_center.shape
        sf = self.downscale_factor

        # --- 2. Downsample inputs to LR ---
        H_lr = self.align_to_multiple(H // sf, 4)
        W_lr = self.align_to_multiple(W // sf, 4)

        input_tensors_lr = [
            F.interpolate(t, size=(H_lr, W_lr), mode='bicubic',
                          align_corners=False, antialias=True).clamp(0., 1.)
            for t in input_tensors
        ]
        noise_map_lr = F.interpolate(noise_map, size=(H_lr, W_lr),
                                     mode='bicubic', align_corners=False,
                                     antialias=True).clamp(0., 1.)

        # --- 3. Build LR input tensor ---
        if self.use_noise_map:
            input_tensor = self.create_input_tensor_with_map(input_tensors_lr, noise_map_lr)
        else:
            input_tensor = torch.cat(input_tensors_lr, dim=1)

        # LR center frame for UNet residual
        lr_center = input_tensors_lr[len(input_tensors_lr) // 2]

        # --- 4. UNet at LR resolution ---
        x0 = self.inc(input_tensor)
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x = self.outc(x0 + x1)

        # UNet residual at LR (same as DenBlock)
        denoised_lr = lr_center - x        # [N, 3, H_lr, W_lr]

        # --- 5. Learned upsample: LR → HR with HR residual ---
        clean_hr = self.learned_upsample(denoised_lr, hr_center)

        return clean_hr

    def create_input_tensor_with_map(self, input_tensors, noise_map):
        input_tensor = None
        for inX in input_tensors:
            if input_tensor is None:
                input_tensor = torch.cat((inX, noise_map), dim=1)
            else:
                input_tensor = torch.cat((input_tensor, inX, noise_map), dim=1)
        return input_tensor
