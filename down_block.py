from torch import nn

from cv_block import CvBlock
from lite_cv_block import LiteCvBlock


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch, simplifed_cv):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            LiteCvBlock(out_ch, out_ch) if simplifed_cv else CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)
