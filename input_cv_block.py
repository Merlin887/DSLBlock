from torch import nn


class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, interm_ch, out_ch, use_noise_map=True, use_depthwise=True):
        super(InputCvBlock, self).__init__()
        self.interm_ch = interm_ch
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * (self.get_input_dimension(use_noise_map)),
                      num_in_frames * self.interm_ch,
                      kernel_size=3,
                      padding=1,
                      groups=self.get_groups_count(use_depthwise, num_in_frames),
                      bias=False),
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.reset_params()

    def get_groups_count(self, use_depthwise, num_in_frames):
        if use_depthwise:
            return num_in_frames
        else:
            return 1

    def get_input_dimension(self, use_noise_map):
        if use_noise_map:
            return 3 + 1
        else:
            return 3

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        return self.convblock(x)
