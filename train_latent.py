import copy
import functools
import os
import time
import glob
import imageio
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from utils import logger
from model.nn import update_ema


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        use_fp16=False,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        active_sh_degree=0,
        white_background=True,
        use_tensorboard=True,
        has_pretrain_weight=False,
        gradient_accumulation_steps=1,
        mem_ratio=1.0,
        args=None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            mixed_precision='fp16' if use_fp16 else 'no',
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=logger.get_dir(),
        )

        options = logger.args_to_dict(args)
        if self.accelerator.is_main_process:
            logger.save_args(options)

        self.model = model
        self.diffusion = diffusion
        self.data = data

        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_fp16 = use_fp16
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.accelerator.num_processes
        self.active_sh_degree = active_sh_degree
        self.has_pretrain_weight = has_pretrain_weight
        self.mem_ratio = mem_ratio

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Call auto-resume during initialization
        if args.auto_resume:
            self.auto_resume()
        
        num_warmup_steps = 1000
        def warmup_lr_schedule(steps):  
            if steps < num_warmup_steps:  
                return float(steps) / float(max(1, num_warmup_steps))  
            return 1.0  
        
        self.warmup_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup_lr_schedule)

        # Prepare everything with accelerator
        self.model, self.opt, self.warmup_scheduler, self.data = self.accelerator.prepare(
            self.model, self.opt, self.warmup_scheduler, self.data
        )

        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and self.accelerator.is_main_process:
            self.writer = logger.Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    def auto_resume(self):
        """Auto-resume from latest checkpoint if available"""
        # Try to auto-resume from latest checkpoint
        exp_dir = os.path.join(logger.get_dir(), "checkpoints")
        if os.path.exists(exp_dir):
            checkpoints = []
            for f in os.listdir(exp_dir):
                if f.startswith("diffusion_") and f.endswith(".pt"):
                    try:
                        step = int(f.replace("diffusion_", "").replace(".pt", ""))
                        checkpoints.append((step, os.path.join(exp_dir, f)))
                    except ValueError:
                        continue
            if len(checkpoints) > 0:
                # Load latest checkpoint
                latest_step, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
                logger.log(f"Auto-resuming from step {latest_step}...")
                self.resume_step = latest_step
                self._load_state_dict(latest_checkpoint, latest_step)

    def _find_resume_step(self, checkpoint_path):
        """Helper function to find step number from checkpoint filename"""
        basename = os.path.basename(checkpoint_path)
        if not basename.startswith("diffusion_"):
            return 0
        try:
            return int(basename.replace("diffusion_", "").replace(".pt", ""))
        except ValueError:
            return 0

    def _load_state_dict(self, checkpoint_path, step):
        """Helper function to load model, optimizer and EMA states"""
        # Load model checkpoint
        model_path = checkpoint_path
        opt_path = os.path.join(logger.get_dir(), "checkpoints", f"opt{step:06d}.pt")
        ema_path = os.path.join(logger.get_dir(), "checkpoints", f"ema_diffusion_0.9999_{step:06d}.pt")
        
        # model_checkpoint = th.load(model_path, map_location=self.accelerator.device)
        opt_checkpoint = th.load(opt_path, map_location=self.accelerator.device)
        ema_checkpoint = th.load(ema_path, map_location=self.accelerator.device)

        # Remove 'module.' prefix from checkpoint keys if present
        model_checkpoint = {k.replace('module.', ''): v for k, v in ema_checkpoint.items()}
        opt_checkpoint = {k.replace('module.', ''): v for k, v in opt_checkpoint.items()}
        
        self.model.load_state_dict(model_checkpoint)
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt.load_state_dict(opt_checkpoint)
    
    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step <= self.lr_anneal_steps
        ):
            batch, model_kwargs = next(self.data)
            self.run_step(batch, model_kwargs)
            if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
         
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, model_kwargs):
        start_time = time.time()
        
        self.forward_backward(batch, model_kwargs)
        self.optimize()
        
        step_time = time.time() - start_time
        logger.logkv_mean("step_time", step_time)
        self.log_step()

    def forward_backward(self, batch, model_kwargs):
        """Training step for diffusion model"""
        micro_latent = batch.to(self.accelerator.device)
        micro_cond = {'cond_images': model_kwargs['cond_images'].to(self.accelerator.device), 'static_latent': model_kwargs['static_latent'].to(self.accelerator.device), 'deformation_position_xyz': model_kwargs['deformation_position_xyz'].to(self.accelerator.device), 'mem_ratio': self.mem_ratio}
        
        with self.accelerator.accumulate(self.model):
            # Get diffusion timesteps
            t, weights = self.schedule_sampler.sample(micro_latent.shape[0], self.accelerator.device)
            
            # Get diffusion losses
            losses, output = self.diffusion.training_losses(
                self.model,
                micro_latent,
                t,
                model_kwargs=micro_cond
            )
            
            loss = losses["loss"].mean()

            log_loss_dict(self.diffusion, t, losses)
            
            if self.use_tensorboard and self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                self.writer.write_dict({k: v.mean().item() for k, v in losses.items()}, self.step)
            
            self.accelerator.backward(loss)

    def optimize(self):
        """Optimization step"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model_params, 1.0)
        
        self.opt.step()
        self.warmup_scheduler.step()
        self.opt.zero_grad()
        
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        # Get scale after optimizer step
        current_scale = self.accelerator.scaler.get_scale()
        logger.logkv("lg_grad_scale", np.log(current_scale))
        
        # Update EMA parameters
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        if not self.accelerator.is_main_process:
            return

        def save_checkpoint(rate, params, name="diffusion"):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {name} {rate}...")
            if not rate:
                filename = f"{name}_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{name}_{rate}_{(self.step+self.resume_step):06d}.pt"
            with open(os.path.join(get_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with open(
            os.path.join(get_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict


def get_logdir():
    p = os.path.join(logger.get_dir(), "checkpoints")
    os.makedirs(p,exist_ok=True)
    return p


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
