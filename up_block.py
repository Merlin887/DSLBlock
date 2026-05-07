from torch import nn

from cv_block import CvBlock
from lite_cv_block import LiteCvBlock


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU) + Upscale'''
    def __init__(self, in_ch, out_ch, simplifed_cv):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            LiteCvBlock(in_ch, in_ch) if simplifed_cv else CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)
