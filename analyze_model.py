from litedvdnet import LiteDVDNet

import torch
import time
from thop import profile

def denoise_sequence(model, seq, noise_map, temp_psz):
    """Denoises a sequence of frames with FastDVDnet.

    Args:
        seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
        noise_map: Tensor. Noise map
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seq.shape
    ctrlfr_idx = int((temp_psz - 1) // 2)
    inframes = list()
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    for fridx in range(numframes):
        # load input frames
        if not inframes:
            # if list not yet created, fill it with temp_patchsz frames
            for idx in range(temp_psz):
                relidx = abs(idx - ctrlfr_idx)  # handle border conditions, reflect
                inframes.append(seq[relidx])
        else:
            del inframes[0]
            relidx = min(fridx + ctrlfr_idx, -fridx + 2 * (numframes - 1) - ctrlfr_idx)  # handle border conditions
            inframes.append(seq[relidx])

        inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz * C, H, W)).to(seq.device)

        # append result to output list
        denframes[fridx] = torch.clamp(model(inframes_t, noise_map), 0., 1.)

    # free memory up
    del inframes
    del inframes_t
    torch.cuda.empty_cache()

    return denframes


def run_model(label, model, H, W, C, num_frames, noise, temp_psz=5):

    net_params = sum(map(lambda x: x.numel(), model.parameters()))

    print(f'Network: {label}, with parameters: {net_params:,d}, '
          f'in_frames: {model.num_input_frames}, '
          f'channels: {model.channels}/{model.channels_2nd}, '
          f'interm channels: {model.interm_ch}, '
          f'simple_cv={model.simple_cv}')

    results = {'label': label, 'resolution': f'{W}x{H}', 'params': net_params}

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            noisy_seq = torch.rand(num_frames, C, H, W).to(device)
            noise_map = torch.FloatTensor([noise / 255]).expand((1, 1, H, W)).to(device)

            tensors_list = list(noisy_seq[0:temp_psz])
            input_frames = torch.stack(tensors_list, dim=0).contiguous().view((1, temp_psz * C, H, W)).to(device)

            print(input_frames.shape)

            macs, params = profile(model, inputs=(input_frames, noise_map),
                                   verbose=False,
                                   ret_layer_info=False,
                                   report_missing=False)

            g_macs = macs / 1024 / 1024 / 1024
            results['gmacs'] = g_macs

            print(f'GMacs: {g_macs:.2f}')

            start_time = time.time()
            denoise_sequence(model=model, seq=noisy_seq, noise_map=noise_map, temp_psz=temp_psz)
            elapsed_time = time.time() - start_time

            results['frame_time_ms'] = elapsed_time / num_frames * 1000
            results['fps'] = num_frames / elapsed_time

            print(f'Single Frame Denoise Time: {results["frame_time_ms"]:.3f} ms')

            return results


def get_models():
    return [
        ('litedvdnet_32_cached',
         LiteDVDNet(num_input_frames=5, inference_mode='Cached',
                    interm_ch=10, simple_cv=True,
                    channels=[32, 64, 128], channels_2nd=[32, 64, 128],
                    use_noise_map=True, use_depthwise=True)),

        ('litedvdnet_downscaled_2_cached',
         LiteDVDNet(num_input_frames=5, inference_mode='DownscaleInBlockCached',
                    interm_ch=10, simple_cv=True,
                    channels=[32, 64, 128], channels_2nd=[32, 64, 128],
                    use_noise_map=True, use_depthwise=True,
                    downscale_factor=2, ds_feat_ch=32)),

        ('litedvdnet_downscaled_3_cached',
         LiteDVDNet(num_input_frames=5, inference_mode='DownscaleInBlockCached',
                    interm_ch=10, simple_cv=True,
                    channels=[32, 64, 128], channels_2nd=[32, 64, 128],
                    use_noise_map=True, use_depthwise=True,
                    downscale_factor=3, ds_feat_ch=32)),

        ('litedvdnet_downscaled_4_32_cached',
         LiteDVDNet(num_input_frames=5, inference_mode='DownscaleInBlockCached',
                    interm_ch=10, simple_cv=True,
                    channels=[32, 64, 128], channels_2nd=[32, 64, 128],
                    use_noise_map=True, use_depthwise=True,
                    downscale_factor=4, ds_feat_ch=32)),

        ('litedvdnet_downscaled_4_64_cached',
         LiteDVDNet(num_input_frames=5, inference_mode='DownscaleInBlockCached',
                    interm_ch=10, simple_cv=True,
                    channels=[32, 64, 128], channels_2nd=[32, 64, 128],
                    use_noise_map=True, use_depthwise=True,
                    downscale_factor=4, ds_feat_ch=64))
    ]

if __name__ == '__main__':

    print('Creating model ...')

    noise = 35
    benchmark_frames = 200

    all_results = []

    models = get_models()
    for label, model in models:
        result = run_model(label, model, H=540, W=960, C=3, num_frames=benchmark_frames, noise=noise, temp_psz=5)
        all_results.append(result)

        # --- Save summary as markdown ---
    md_path = 'benchmark_summary.md'
    with open(md_path, 'w') as f:
        f.write(f'# LiteDVDNet Benchmark Summary\n\n')
        f.write(f'Noise: {noise}, Benchmark: {benchmark_frames} frames\n\n')
        f.write('| Model | Resolution | Params | GMACs | ms/frame | FPS |\n')
        f.write('|-------|-----------|-------:|------:|---------:|----:|\n')
        for r in all_results:
            f.write(f'| {r["label"]} | {r["resolution"]} | {r["params"]:,d} | {r["gmacs"]:.2f} '
                    f'| {r["frame_time_ms"]:.0f} | {r["fps"]:.0f} |\n')

    print(f'\nSummary saved to {md_path}')










