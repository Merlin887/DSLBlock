import importlib
import os
import shutil
import time
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataloaders import train_dali_loader
from dataset import ValDataset
from train_common import resume_training, save_model_checkpoint, need_ortog
from utils import (
    normalize_augment, init_logging, svd_orthogonalization,
    close_logger, load_options, batch_psnr, set_random_seed
)

@dataclass
class TrainingConfig:

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-8

    # Gradient settings
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Learning rate scheduler
    scheduler_type: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.01

    # Loss function
    loss_type: str = 'mse'
    perceptual_weight: float = 0.1

    # Validation
    validate_every_n_epochs: int = 1
    val_noise_levels: list = field(default_factory=lambda: [15, 25, 50])

    # Logging
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 1

    # Paths
    trainset_dir: str = ''
    valset_dir: str = ''
    log_dir: str = 'experiments'

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation metric and stops training if no improvement
    for a specified number of epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig, steps_per_epoch: int):

    if config.scheduler_type == 'step':
        # Step-based scheduler
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20, 40, 60],
            gamma=0.5
        )

    elif config.scheduler_type == 'cosine':
        # Cosine annealing - smooth decay
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")



class TrainRunner:

    def __init__(self, options_path: str):
        self.options_path = options_path
        self.options = self.get_options(options_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Will be initialized in train()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        self.early_stopping = None

        # Tracking
        self.best_psnr = 0.0
        self.global_step = 0

    def get_options(self, options_path: str) -> Dict:
        """Parse and normalize configuration options."""
        options = load_options(options_path)

        # Set defaults for all training options
        options.setdefault('manual_seed', 42)
        options.setdefault('use_cuda', True)
        options.setdefault('batch_size', 64)
        options.setdefault('resume_training', False)
        options.setdefault('resume_dir', '')
        options.setdefault('lr', 0.001)
        options.setdefault('save_every', 100)
        options.setdefault('save_ckpt_every_epochs', 10)
        options.setdefault('noise_ival', [5, 55])
        options.setdefault('val_noiseL', 25.0)
        options.setdefault('trainset_dir', './datasets/davis2017/videos/480p')
        options.setdefault('valset_dir', './datasets/Set8/validation')
        options.setdefault('use_amp', True)
        options.setdefault('gradient_clip_norm', 1.0)
        options.setdefault('gradient_accumulation_steps', 1)
        options.setdefault('early_stopping_patience', 40)
        options.setdefault('early_stopping_min_delta', 0.001)
        options.setdefault('loss_type', 'mse')
        options.setdefault('weight_decay', 1e-8)
        options.setdefault('scheduler_type', 'cosine')
        options.setdefault('warmup_epochs', 5)

        # Derive temp_patch_size from model_params.num_input_frames
        options['temp_patch_size'] = options['model_params']['num_input_frames']

        # Normalize noise levels to [0, 1]
        options['val_noiseL'] /= 255.0
        options['noise_ival'][0] /= 255.0
        options['noise_ival'][1] /= 255.0

        return options

    def create_model(self) -> nn.Module:
        """Dynamically load and instantiate model."""
        module = importlib.import_module(self.options['module'])
        model_class = getattr(module, self.options['model_name'])
        return model_class(**self.options['model_params'])

    def create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.options.get('loss_type', 'mse')

        if loss_type == 'mse':
            return nn.MSELoss(reduction='sum')
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def warmup_lr(self, epoch: int, warmup_epochs: int = 5) -> float:
        """Linear warmup for learning rate."""
        if epoch < warmup_epochs:
            return self.options['lr'] * (epoch + 1) / warmup_epochs
        return self.options['lr']


    def train_step(self, batch_data: torch.Tensor, criterion: nn.Module, ctrl_fr_idx: int, accumulation_step: int) -> float:
        args = self.options

        # Prepare data
        img_train, gt_train = normalize_augment(batch_data, ctrl_fr_idx)
        N, _, H, W = img_train.size()

        # Generate noise
        stdn = torch.empty((N, 1, 1, 1), device=self.device).uniform_(
            args['noise_ival'][0], args['noise_ival'][1]
        )

        noise = torch.randn_like(img_train) * stdn
        imgn_train = (img_train + noise).to(self.device, non_blocking=True)
        gt_train = gt_train.to(self.device, non_blocking=True)
        noise_map = stdn.expand((N, 1, H, W))

        with autocast(enabled=args.get('use_amp', True)):
            out_train = self.model(imgn_train, noise_map)
            loss = criterion(out_train, gt_train)

            # Scale loss for gradient accumulation
            loss = loss / args.get('gradient_accumulation_steps', 1)

        self.scaler.scale(loss).backward()

        # Only update weights after accumulation
        if (accumulation_step + 1) % args.get('gradient_accumulation_steps', 1) == 0:

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                args.get('gradient_clip_norm', 1.0)
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()


        return loss.item() * args.get('gradient_accumulation_steps', 1)

    @torch.no_grad()
    def validate(self, dataset_val, noise_levels: list = None) -> Dict[str, float]:
        if noise_levels is None:
            noise_levels = [self.options['val_noiseL']]

        self.model.eval()

        results = {}

        for noise_level in noise_levels:
            psnr_sum = 0.0

            for seq_val in dataset_val:
                noise = torch.randn_like(seq_val) * noise_level
                seqn_val = (seq_val + noise).to(self.device)

                noise_map = torch.full(
                    (1, 1, seq_val.shape[-2], seq_val.shape[-1]),
                    noise_level,
                    device=self.device
                )

                out_val = self.denoise_sequence(
                    seq=seqn_val,
                    noise_map=noise_map,
                    temp_psz=self.options['temp_patch_size']
                )

                out_cpu = out_val.cpu()
                psnr_sum += batch_psnr(out_cpu, seq_val.squeeze_(), 1.0)

            n_samples = len(dataset_val)
            noise_key = f"sigma_{int(noise_level * 255)}"
            results[f'psnr_{noise_key}'] = psnr_sum / n_samples

        self.model.train()

        return results

    @torch.no_grad()
    def denoise_sequence(
        self,
        seq: torch.Tensor,
        noise_map: torch.Tensor,
        temp_psz: int
    ) -> torch.Tensor:

        numframes, C, H, W = seq.shape
        ctrlfr_idx = (temp_psz - 1) // 2
        denframes = torch.empty((numframes, C, H, W), device=seq.device)

        # Pre-allocate frame buffer
        frame_buffer = torch.empty((temp_psz, C, H, W), device=seq.device)

        for fridx in range(numframes):
            # Fill frame buffer with appropriate frames
            for buf_idx in range(temp_psz):
                # Calculate source frame index with reflection padding
                src_idx = fridx - ctrlfr_idx + buf_idx
                if src_idx < 0:
                    src_idx = -src_idx
                elif src_idx >= numframes:
                    src_idx = 2 * (numframes - 1) - src_idx
                frame_buffer[buf_idx] = seq[src_idx]

            # Reshape for model input
            input_tensor = frame_buffer.view(1, temp_psz * C, H, W)

            # Denoise
            denframes[fridx] = torch.clamp(
                self.model(input_tensor, noise_map), 0.0, 1.0
            )

        return denframes

    def save_best_model(self, psnr: float, epoch: int):
        save_path = os.path.join(
            self.options['log_dir'],
            f"{self.options['model_description']}_best.pth"
        )

        state_dict = self.model.module.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'psnr': psnr,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def train(self):

        args = self.options

        # Set random seed
        set_random_seed(args.get('manual_seed', 42))

        # Load datasets
        print('> Loading datasets...')
        dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
        loader_train = train_dali_loader(
            batch_size=args['batch_size'],
            file_root=args['trainset_dir'],
            sequence_length=args['temp_patch_size'],
            crop_size=args['patch_size'],
            epoch_size=args['max_number_patches'],
            random_shuffle=True,
            temp_stride=3
        )

        # Setup model
        torch.backends.cudnn.benchmark = True
        self.model = self.create_model()
        args['model_description'] = self.model.get_desciption()
        self.model = nn.DataParallel(self.model, device_ids=[0]).to(self.device)

        # Setup logging directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args['log_dir'] = os.path.join('experiments', f"{args['model_description']}_{timestamp}")
        os.makedirs(args['log_dir'], exist_ok=True)

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=args['log_dir'])
        logger = init_logging(args)

        # Copy config
        shutil.copy(self.options_path,
                   os.path.join(args['log_dir'], f"{args['model_description']}.yaml"))

        # Setup loss function
        criterion = self.create_loss_function().to(self.device)

        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args['lr'],
            weight_decay=args.get('weight_decay', 1e-8),
            betas=(0.9, 0.999)
        )

        training_config = TrainingConfig(**{k: args.get(k, getattr(TrainingConfig, k, None))
                                                          for k in TrainingConfig.__dataclass_fields__})

        # Setup scheduler
        num_minibatches = int(args['max_number_patches'] // args['batch_size'])
        self.scheduler = create_scheduler(self.optimizer, training_config, num_minibatches)

        self.scaler = GradScaler(enabled=args.get('use_amp', True))

        self.early_stopping = EarlyStopping(
            patience=args.get('early_stopping_patience', 15),
            min_delta=args.get('early_stopping_min_delta', 0.01),
            mode='max'
        )

        # Resume if needed
        start_epoch, training_params = resume_training(args, self.model, self.optimizer)
        self.global_step = training_params.get('step', 0)

        # Training info
        ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
        logger.info(f'Training with improvements:')
        logger.info(f'  - Mixed Precision: {args.get("use_amp", True)}')
        logger.info(f'  - Gradient Accumulation: {args.get("gradient_accumulation_steps", 1)}')
        logger.info(f'  - Loss: {args.get("loss_type", "mse")}')
        logger.info(f'  - Scheduler: {args.get("scheduler_type", "cosine")}')
        logger.info(f'  - Early Stopping: patience={args.get("early_stopping_patience", 15)}, min_delta={args.get("early_stopping_min_delta", 0.01)}')

        # Main training loop
        start_time = time.time()

        for epoch in range(start_epoch, args['epochs'] + 1):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            # Update orthogonalization status for this epoch
            training_params['orthog_enabled'] = need_ortog(epoch, args)

            if epoch <= args.get('warmup_epochs', 5):
                warmup_lr = self.warmup_lr(epoch, args.get('warmup_epochs', 5))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\n[Epoch {epoch}] LR: {current_lr:.2e}, Orthog: {training_params["orthog_enabled"]}')

            # Training epoch
            self.model.train()
            self.optimizer.zero_grad()

            for i, data in enumerate(loader_train):
                loss = self.train_step(data[0]['data'], criterion, ctrl_fr_idx, i)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % args.get('save_every', 100) == 0:
                    # SVD orthogonalization
                    if training_params.get('orthog_enabled', False):
                        self.model.apply(svd_orthogonalization)

                    avg_loss = epoch_loss / num_batches
                    print(f'[Epoch {epoch}][{i+1}/{num_minibatches}] Loss: {loss:.4f} (avg: {avg_loss:.4f})')

                    # TensorBoard logging
                    self.writer.add_scalar('Loss/train', loss, self.global_step)
                    self.writer.add_scalar('LR', current_lr, self.global_step)

            # Update scheduler
            if epoch > args.get('warmup_epochs', 5):
                self.scheduler.step()

            # Validation
            val_results = self.validate(dataset_val)
            main_psnr = list(val_results.values())[0]

            # Log validation results
            print(f'[Epoch {epoch}] Validation Results:')
            for metric_name, value in val_results.items():
                print(f'  {metric_name}: {value:.4f}')
                self.writer.add_scalar(f'Val/{metric_name}', value, epoch)

            logger.info(f'[Epoch {epoch}] PSNR: {main_psnr:.4f}')

            # Save best model
            if main_psnr > self.best_psnr:
                self.best_psnr = main_psnr
                self.save_best_model(main_psnr, epoch)
                print(f'  ★ New best model! PSNR: {main_psnr:.4f}')

            # Save regular checkpoint
            training_params['start_epoch'] = epoch
            training_params['step'] = self.global_step
            save_model_checkpoint(self.model, args, self.optimizer, training_params, epoch)

            # Early stopping check
            if self.early_stopping(main_psnr):
                print(f'\nEarly stopping triggered after {epoch} epochs')
                logger.info(f'Early stopping at epoch {epoch}')
                break

            # Time estimation
            epoch_time = time.time() - epoch_start_time
            remaining = (args['epochs'] - epoch) * epoch_time
            print(f'Epoch time: {self._format_time(epoch_time)} | ETA: {self._format_time(remaining)}')

        # Training complete
        total_time = time.time() - start_time
        print(f'\nTraining complete! Total time: {self._format_time(total_time)}')
        print(f'Best PSNR: {self.best_psnr:.4f}')

        # Save final results
        self.save_results(args['model_description'], args['log_dir'], logger)

        # Cleanup
        self.writer.close()
        close_logger(logger)

    def save_results(self, model_desc: str, log_dir: str, logger):

        # Use same folder name as in experiments
        log_dir_name = os.path.basename(log_dir)
        save_dir = os.path.join('models', log_dir_name)

        os.makedirs(save_dir, exist_ok=True)

        # Copy artifacts
        for filename in [f'{model_desc}.yaml', f'{model_desc}.pth',
                         f'{model_desc}_best.pth', 'log.txt']:
            src = os.path.join(log_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(save_dir, filename))

        # Copy TensorBoard logs
        tensorboard_dest = os.path.join(save_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_dest, exist_ok=True)

        for filename in os.listdir(log_dir):
            if filename.startswith('events.out.tfevents'):
                src = os.path.join(log_dir, filename)
                shutil.copy(src, os.path.join(tensorboard_dest, filename))
                logger.info(f'Copied TensorBoard log: {filename}')

        logger.info(f'Results saved to {save_dir}')

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        return time.strftime("%H:%M:%S", time.gmtime(seconds))