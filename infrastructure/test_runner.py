import importlib
import os
import pprint
import shutil
import time
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from infrastructure.test_results import TestCaseResult, organize_results, get_test_case_table
from utils import batch_psnr, init_logger_test, \
    variable_to_cv2_image, open_sequence, close_logger, get_current_time, create_folder, \
    save_as_json, load_options, apply_padding, remove_padding, create_video_from_images, calculate_strred, \
    calculate_ssim


# =============================================================================
# Comparison Report Generator (Markdown)
# Produces *_comparison.md in the same format as test_runner_improved.py
# =============================================================================
class ComparisonReportGenerator:
    """
    Generate a compact comparison markdown table identical to the one produced
    by ReportGenerator.comparison_table_markdown in test_runner_improved.

    Layout: models as rows, one column per noise level with PSNR / STRRED / SSIM,
    plus a frame-time column.
    """

    @staticmethod
    def _build_comparison_data(results: List[TestCaseResult]):
        """
        Group valid results by (model, noise_level) and by model.
        Returns (valid, model_names, noise_levels, grouped, by_model) or None.
        """
        valid = [r for r in results if r.psnr_clean > 0]
        if not valid:
            return None

        # Preserve insertion order for model names and noise levels
        model_names = list(dict.fromkeys(r.model_name for r in valid))
        noise_levels = list(dict.fromkeys(r.noise_level for r in valid))

        grouped: Dict[Tuple[str, int], List[TestCaseResult]] = defaultdict(list)
        for r in valid:
            grouped[(r.model_name, r.noise_level)].append(r)

        by_model: Dict[str, List[TestCaseResult]] = defaultdict(list)
        for r in valid:
            by_model[r.model_name].append(r)

        return valid, model_names, noise_levels, grouped, by_model

    @staticmethod
    def _format_metrics_cell(psnr: float, strred: float, ssim: float) -> str:
        """Format PSNR / STRRED / SSIM into a single cell string."""
        return f"{psnr:.2f} / {strred:.4f} / {ssim:.4f}"

    @staticmethod
    def generate_comparison_md(results: List[TestCaseResult], path: str):
        """
        Write the *_comparison.md file.

        Format matches ReportGenerator.comparison_table_markdown from
        test_runner_improved: one row per model, one column per noise level
        containing averaged PSNR / STRRED / SSIM, and a final frame-time column.
        """
        if not results:
            return

        data = ComparisonReportGenerator._build_comparison_data(results)
        if data is None:
            return

        valid, model_names, noise_levels, grouped, by_model = data

        lines = [
            "# Model Comparison",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        ]

        # Header row
        header_parts = ["Model"]
        for nl in noise_levels:
            header_parts.append(f"σ={nl} (PSNR / STRRED / SSIM)")
        header_parts.append("Frame time (ms)")
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join("---" for _ in header_parts) + "|")

        # One row per model
        for m in model_names:
            row_parts = [m]
            for nl in noise_levels:
                vals = grouped.get((m, nl), [])
                if vals:
                    avg_psnr = statistics.mean(r.psnr_clean for r in vals)
                    avg_strred = statistics.mean(r.strred_score for r in vals)
                    avg_ssim = statistics.mean(r.ssim_score for r in vals)
                    row_parts.append(
                        ComparisonReportGenerator._format_metrics_cell(avg_psnr, avg_strred, avg_ssim)
                    )
                else:
                    row_parts.append("N/A")

            all_vals = by_model.get(m, [])
            avg_time = statistics.mean(r.single_frame_time for r in all_vals) if all_vals else 0
            row_parts.append(f"{avg_time:.2f}")
            lines.append("| " + " | ".join(row_parts) + " |")

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


class TestRunner:
    def __init__(self, options_path: str):
        self.options_path = options_path
        self.options = load_options(options_path)
        self.test_settings =  self.options['test_settings']
        self.suite_path = self.get_path()
        create_folder(self.suite_path)
        self.logger = init_logger_test(self.suite_path)

    def get_path(self) -> str:
        now = datetime.now().strftime("%m%d%Y_%H%M%S")
        save_path = self.options['test_settings']['save_path']
        return str(os.path.join(save_path, self.options['id'] + '_' + now))

    def log(self, message: str):
        self.logger.info(message)

    def close_logger(self):
        close_logger(self.logger)

    def get_device(self) -> torch.device:

        # Sets data type according to CPU or GPU modes
        if self.options['test_settings']['use_cuda']:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    def create_model(self, model) -> nn.Module:

        module = importlib.import_module(model['module'])
        model_class = getattr(module, model['model_name'])
        instance = model_class(**model['model_params'])

        return instance


    def get_models_to_compare(self):
        # Load models to compare
        all_models_dir = self.options['models_to_compare']

        models_to_compare = []
        for model_dir in os.listdir(all_models_dir):
            models_metadata = {}
            model_opts = load_options(os.path.join(all_models_dir, model_dir, f'{model_dir}.yaml'))
            models_metadata['model_name'] = model_opts['model_name']
            models_metadata['description'] = model_dir
            models_metadata['module'] = model_opts['module']
            models_metadata['model_params'] = model_opts['model_params']

            if 'inference_mode' in model_opts['model_params']:
                models_metadata['model_params']['inference_mode'] = model_opts['model_params']['inference_mode']

            pretrain_ckpt = os.path.join(all_models_dir, model_dir, f'{model_dir}.pth')
            models_metadata['model_params']['pretrain_ckpt'] = pretrain_ckpt
            models_to_compare.append(models_metadata)

        return models_to_compare

    def run(self):

        test_suite_id = self.options['id']
        options_copy_path = os.path.join(self.suite_path, f'{test_suite_id}.yml')
        shutil.copy(self.options_path, options_copy_path)

        self.log(f"### Starting test runner at {get_current_time()} ###")
        self.log(pprint.pformat(self.options))

        test_case_results = []

        models_to_compare = self.get_models_to_compare()
        noise_sigmas = self.options['test_settings']['noise_sigma']

        for noise_sigma in noise_sigmas:

            self.log(f"\nRunning test case for noise level: {noise_sigma}\n")

            for model_metadata in models_to_compare:
                model_name = model_metadata['model_name']
                model_description = model_metadata['description']
                input_frames = model_metadata['model_params']['num_input_frames']
                self.log(f"Running model: {model_name} ({model_description})")

                # create model folder for experiments
                model_folder = os.path.join(self.suite_path, model_description)
                create_folder(model_folder)

                for test_case in self.options['test_cases']:
                    # create model
                    model = self.create_model(model_metadata)
                    test_case['noise_sigma'] = noise_sigma
                    self.log(f"Running test case: {test_case}")
                    result = self.run_model(test_case, model, model_description, input_frames, model_folder)
                    test_case_results.append(result)
                    del model

        self.log(f"### Test completed at {get_current_time()} ###")

        save_path = os.path.join(self.suite_path, f'{test_suite_id}_Results.json')
        test_suite_results = organize_results(test_case_results)
        self.log('\n' + get_test_case_table(test_case_results))
        save_as_json(test_suite_results, save_path)

        md_path = os.path.join(self.suite_path, f'{test_suite_id}_comparison.md')
        ComparisonReportGenerator.generate_comparison_md(test_case_results, md_path)
        self.log(f"Saved Markdown comparison report: {md_path}")

        # close logger
        self.close_logger()


    def run_model(self, test_case, loaded_model, model_description: str, input_frames: int, model_folder: str) -> TestCaseResult:

        experiment_folder = os.path.join(model_folder, f"{test_case['id']}_{test_case['noise_sigma']}")
        create_folder(experiment_folder)

        device = self.get_device()
        loaded_model.to(self.get_device())

        net_params = sum(map(lambda x: x.numel(), loaded_model.parameters()))
        self.log(f'Network: {model_description}, with parameters: {net_params:,d}. Window size: {input_frames}')

        # Sets the model in evaluation mode (e.g. it removes BN)
        loaded_model.eval()

        with torch.no_grad():
            # process data
            with torch.cuda.amp.autocast(True):
                original_seq, loadtime = self.load_sequence(test_case, device)
                noisy_seq, denoised_seq, runtime = self.denoise_sequence(original_seq, loaded_model,
                                                                         input_frames, test_case, device)

        strred_score = 0
        ssim_score = 0
        psnr = batch_psnr(denoised_seq, original_seq, 1., False)
        psnr_noisy = batch_psnr(noisy_seq.squeeze(), original_seq, 1., False)
        seq_length = original_seq.size()[0]
        average_frame_time = runtime / seq_length

        self.log(f"Finished denoising {test_case['test_data_path']}")
        self.log(f"\tDenoised {seq_length} frames in {runtime:.3f}s, loaded seq in {loadtime:.3f}s")
        self.log(f"\tSingle frame denoising time {round(average_frame_time, 3) * 1000} msec")
        self.log(f"\tPSNR noisy {psnr_noisy:.4f}dB, PSNR result {psnr:.4f}dB")

        # Save outputs
        if self.options['test_settings']['save_results']:

            original_data_folder = test_case['test_data_path']
            filename = os.path.basename(test_case['test_data_path'])
            # Save sequence
            self.save_out_seq(noisy_seq = noisy_seq,
                              denoised_seq = denoised_seq,
                              save_dir=experiment_folder,
                              filename=filename,
                              fext=self.test_settings['suffix'],
                              save_noisy=self.test_settings['save_noisy'])

            strred_score = 0.0
            frames = self.test_settings['max_num_fr_per_seq']
            orig_video_path = create_video_from_images(original_data_folder,f'{filename}_original.mp4', experiment_folder,frames, 30)
            denoised_video_path = create_video_from_images(experiment_folder,f'{filename}_denoised.mp4',experiment_folder,frames, 30)
            if self.options['test_settings']['calculate_strred']:
                strred_score = calculate_strred(orig_video_path, denoised_video_path, frames).item()
                self.log(f'ST-RRED score: {strred_score}')

            ssim_score = 0.0
            if self.options['test_settings']['calculate_ssim']:
                ssim_score_array = calculate_ssim(orig_video_path, denoised_video_path, frames)
                ssim_score = np.mean(np.array(ssim_score_array)).item()
                self.log(f'SSIM score: {ssim_score}')



        tc_result = TestCaseResult(model_name=model_description,
                              test_case_name=test_case['id'],
                              noise_level=test_case['noise_sigma'],
                              strred_score=strred_score,
                              ssim_score=ssim_score,
                              psnr_noisy=round(psnr_noisy, 3),
                              psnr_clean=round(psnr, 3),
                              total_denoising_time=round(runtime, 3) * 1000,
                              single_frame_time=round(average_frame_time, 3) * 1000)

        return tc_result

    def load_sequence(self, test_case, device):

        start_time = time.time()

        # process data
        original_seq, _, _ = open_sequence(test_case['test_data_path'],
                                           self.test_settings['gray'],
                                           expand_if_needed=False,
                                           max_num_fr=self.test_settings['max_num_fr_per_seq'])

        original_seq = torch.from_numpy(original_seq).to(device)
        loading_time = time.time() - start_time

        return  original_seq, loading_time

    def denoise_sequence(self, original_seq, model_temp, temp_patch_size, test_case, device):

        noise_level = test_case['noise_sigma'] / 255

        # Add noise
        noise = torch.empty_like(original_seq).normal_(mean=0, std=noise_level).to(device)
        noisy_seq = original_seq + noise
        noisestd = torch.FloatTensor([noise_level]).to(device)

        numframes, C, H, W = noisy_seq.shape
        noise_map = noisestd.expand((1, 1, H, W))
        padded_noisyseq, padded_noisemap = apply_padding(noisy_seq, noise_map)

        start_time = time.time()

        denoised_seq = self.denoise(model=model_temp,
                                    seq=padded_noisyseq,
                                    noise_map=padded_noisemap,
                                    temp_psz=temp_patch_size)

        denoising_time = time.time() - start_time

        denoised_seq = remove_padding(original_seq, denoised_seq)
        return noisy_seq, denoised_seq, denoising_time

    def save_out_seq(self, noisy_seq, denoised_seq, save_dir, filename, fext, save_noisy):
        """Saves the denoised and noisy sequences under save_dir
        """
        seq_len = noisy_seq.size()[0]
        for idx in range(seq_len):
            # Build Outname
            noisy_name = os.path.join(save_dir, f'{filename}_noisy_{idx:04d}{fext}')
            denoised_name = os.path.join(save_dir, f'{filename}_denoised_{idx:04d}{fext}')

            # Save result
            if save_noisy:
                noisyimg = variable_to_cv2_image(noisy_seq[idx].clamp(0., 1.))
                cv2.imwrite(noisy_name, noisyimg)

            outimg = variable_to_cv2_image(denoised_seq[idx].unsqueeze(dim=0))
            cv2.imwrite(denoised_name, outimg)

    def denoise(self, model, seq, noise_map, temp_psz):
        r"""Denoises a sequence of frames with model provided.

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

        # convert to appropiate type and return
        return denframes
