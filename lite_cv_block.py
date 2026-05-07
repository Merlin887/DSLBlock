from torch import nn


class LiteCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(LiteCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
