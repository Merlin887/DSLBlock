import torch
from torch import nn
from enum import Enum

from down_block import DownBlock
from dsl_block import DSLBlock
from input_cv_block import InputCvBlock
from output_cv_block import OutputCvBlock
from up_block import UpBlock


class InferenceMode(Enum):
    Basic = 0
    Cached = 1
    DownscaleInBlock = 2
    DownscaleInBlockCached = 3


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, channels, interm_ch, simp_cv, num_input_frames=3,
                 use_noise_map=True, use_depthwise=True):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = channels[0]
        self.chs_lyr1 = channels[1]
        self.chs_lyr2 = channels[2]
        self.use_noise_map = use_noise_map

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

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input_tensors: list, noise_map):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        if (self.use_noise_map):
            input_tensor = self.create_input_tensor_with_map(input_tensors, noise_map)
        else:
            input_tensor = torch.cat(input_tensors, dim=1)

        denoisedFrame = input_tensors[len(input_tensors) // 2]

        x0 = self.inc(input_tensor)
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)
        # Residual
        x = denoisedFrame - x

        return x

    def create_input_tensor_with_map(self, input_tensors, noise_map):
        input_tensor = None

        for inX in input_tensors:
            if input_tensor is None:
                input_tensor = torch.cat((inX, noise_map), dim=1)
            else:
                input_tensor = torch.cat((input_tensor, inX, noise_map), dim=1)

        return input_tensor


class LiteDVDNet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def is_dowscaled_mode(self):
        return self.inference_mode == InferenceMode.DownscaleInBlock or self.inference_mode == InferenceMode.DownscaleInBlockCached


    def __init__(self, num_input_frames=5, inference_mode='Basic', interm_ch=30, simple_cv=False,
                 channels=[32, 64, 128], channels_2nd = [32, 64, 128],
                 use_noise_map=True, use_depthwise=True,
                 pretrain_ckpt=None, downscale_factor=2, ds_feat_ch=32):
        super(LiteDVDNet, self).__init__()
        self.num_input_frames = num_input_frames

        self.inference_mode = InferenceMode[inference_mode]
        self.simple_cv = simple_cv
        self.interm_ch = interm_ch
        self.use_noise_map = use_noise_map
        self.use_depthwise = use_depthwise
        self.downscale_factor = downscale_factor
        self.ds_feat_ch = ds_feat_ch

        self.prev_den_frame = None
        self.current_den_frame = None
        self.future_den_frame = None

        self.channels = channels
        self.channels_2nd = channels_2nd

        if self.is_dowscaled_mode() :
            denoising_block = DSLBlock
            downscaling_settings = dict(downscale_factor=downscale_factor, upsample_feat_ch=ds_feat_ch)
        else:
            denoising_block = DenBlock
            downscaling_settings = {}

        self.temp1 = denoising_block(channels,
                                     interm_ch,
                                     simple_cv,
                                     num_input_frames=3,
                                     use_noise_map = use_noise_map,
                                     use_depthwise = use_depthwise,
                                     **downscaling_settings)

        self.temp2 = denoising_block(channels_2nd,
                                     interm_ch,
                                     simple_cv,
                                     num_input_frames=3,
                                     use_noise_map = use_noise_map,
                                     use_depthwise = use_depthwise,
                                     **downscaling_settings)

        # Init weights
        self.reset_params()

        if pretrain_ckpt is not None:
            self.load(pretrain_ckpt)

    def get_desciption(self):
        chs_lyr0 = self.channels[0]
        chs_lyr1 = self.channels[1]
        chs_lyr2 = self.channels[2]
        chs_lyr0_2nd = self.channels_2nd[0]
        chs_lyr1_2nd = self.channels_2nd[1]
        chs_lyr2_2nd = self.channels_2nd[2]
        simple_cv = "_s" if self.simple_cv else ""
        interm_ch = self.interm_ch
        use_map = "_with_map" if self.use_noise_map else "_no_map"
        use_depthwise = "_with_depthwise" if self.use_depthwise else ""
        if self.inference_mode == InferenceMode.DownscaleInBlock or self.inference_mode == InferenceMode.DownscaleInBlockCached:
            ds = f"_dsib{self.downscale_factor}f{self.ds_feat_ch}"
        return (f'{__class__.__name__.lower()}[{chs_lyr0}_{chs_lyr1}_{chs_lyr2}]'
                f'[{chs_lyr0_2nd}_{chs_lyr1_2nd}_{chs_lyr2_2nd}]_'
                f'ich{interm_ch}{simple_cv}{use_map}{use_depthwise}{ds}')

    def load(self, pretrain_ckpt):
        state_temp_dict = torch.load(pretrain_ckpt)

        # Handle checkpoint format from ImprovedTrainRunner (wrapped with metadata)
        if isinstance(state_temp_dict, dict) and 'model_state_dict' in state_temp_dict:
            state_temp_dict = state_temp_dict['model_state_dict']

        # Try stripping DataParallel "module." prefix; if nothing matches, use as-is
        state_dict = self.extract_dict(state_temp_dict, string_name="module.")
        if not state_dict:
            state_dict = state_temp_dict

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [load] Missing keys (will be randomly initialized): "
                  f"{[k.split('.')[0] for k in missing if '.' in k]}")
        if unexpected:
            print(f"  [load] Unexpected keys (ignored): {unexpected}")

    def extract_dict(self, ckpt_state, string_name, replace_name=''):
        m_dict = {}
        for k, v in ckpt_state.items():
            if string_name in k:
                m_dict[k.replace(string_name, replace_name)] = v
        return m_dict

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        match(self.inference_mode):
            case InferenceMode.Basic:
                return self.forward_basic(x, noise_map)
            case InferenceMode.Cached:
                return self.forward_cached(x, noise_map)
            case InferenceMode.DownscaleInBlock:
                return self.forward_basic(x, noise_map)
            case InferenceMode.DownscaleInBlockCached:
                return self.forward_cached(x, noise_map)

    def forward_basic(self, x, noise_map):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1([x0, x1, x2], noise_map)
        x21 = self.temp1([x1, x2, x3], noise_map)
        x22 = self.temp1([x2, x3, x4], noise_map)

        # Second stage
        x = self.temp2([x20, x21, x22], noise_map)

        return x

    @staticmethod
    def _align_to_multiple(value: int, multiple: int) -> int:
        '''Round *up* to the nearest multiple (ensures UNet compatibility).'''
        return ((value + multiple - 1) // multiple) * multiple


    def forward_cached(self, x, noise_map):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames))

        # Denoise prev frame and buffer it, or get from buffer
        if self.prev_den_frame is None:
            x20 = self.temp1([x0, x1, x2], noise_map)
            self.prev_den_frame = x20
        else:
            x20 = self.prev_den_frame

        # Denoise current frame and buffer it, or get from buffer
        if self.current_den_frame is None:
            x21 = self.temp1([x1, x2, x3], noise_map)
            self.current_den_frame = x21
        else:
            x21 = self.current_den_frame

        # Denoise future frame and buffer it, or get from buffer
        if self.future_den_frame is None:
            x22 = self.temp1([x2, x3, x4], noise_map)
            self.future_den_frame = x22
        else:
            x22 = self.future_den_frame

        # Second stage
        x = self.temp2([x20, x21, x22], noise_map)

        # Shift all buffers
        self.prev_den_frame = self.current_den_frame
        self.current_den_frame = self.future_den_frame
        self.future_den_frame = None


        return x