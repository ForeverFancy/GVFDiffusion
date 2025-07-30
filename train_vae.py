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
import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import pytorch3d.ops

from utils import logger
from utils.loss_util import ssim
from utils.lpips.lpips import LPIPS
from model.nn import update_ema


class TrainLoop:
    def __init__(
        self,
        model,
        static_vae,
        static_vae_backbone,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_tensorboard=True,
        render_lpips_weight=1.0,
        render_l1_weight=1.0,
        render_ssim_weight=0.2,
        has_pretrain_weight=False,
        xyz_loss_weight=0.1,
        knn_k=8,
        beta=7.0,
        kl_weight=1e-5,
        gradient_accumulation_steps=1,
        static_vae_steps=5000,
        args=None,
        auto_resume=False,
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
        self.static_vae = static_vae
        self.static_vae_backbone = static_vae_backbone
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
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.accelerator.num_processes

        self.has_pretrain_weight = has_pretrain_weight
        self.knn_k = knn_k
        self.beta = beta
        self.xyz_loss_weight = xyz_loss_weight
        self.kl_weight = kl_weight
        self.static_vae_steps = static_vae_steps
        self.interpolation_loss_func = compute_interpolation_loss_delta_interp

        self.vgg = LPIPS(net_type='vgg').eval().to(self.accelerator.device)
        self.L1loss = th.nn.L1Loss()
        self.render_lpips_weight = render_lpips_weight
        self.render_l1_weight = render_l1_weight
        self.render_ssim_weight = render_ssim_weight

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        self.static_vae_params = [params for params in self.static_vae_backbone.parameters() if params.requires_grad]
        self.static_vae_master_params = self.static_vae_params
        self.static_vae_opt = AdamW(self.static_vae_master_params, lr=self.lr*0.1, weight_decay=self.weight_decay)

        if auto_resume:
            self.auto_resume()
        
        num_warmup_steps = 1000
        def warmup_lr_schedule(steps):  
            if steps < num_warmup_steps:  
                return float(steps) / float(max(1, num_warmup_steps))  
            return 1.0  
        
        self.warmup_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup_lr_schedule)

        # Prepare everything with accelerator
        self.model, self.static_vae, self.static_vae_backbone, self.opt, self.static_vae_opt, self.warmup_scheduler, self.data = self.accelerator.prepare(
            self.model, self.static_vae, self.static_vae_backbone, self.opt, self.static_vae_opt, self.warmup_scheduler, self.data,
        )

        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]
        self.ema_params_static_vae = [
            copy.deepcopy(self.static_vae_master_params) for _ in range(len(self.ema_rate))
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
                if f.startswith("deformation_") and f.endswith(".pt"):
                    try:
                        step = int(f.replace("deformation_", "").replace(".pt", ""))
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
        if not basename.startswith("deformation_"):
            return 0
        try:
            return int(basename.replace("deformation_", "").replace(".pt", ""))
        except ValueError:
            return 0

    def _load_state_dict(self, checkpoint_path, step):
        """Helper function to load model, optimizer and EMA states"""
        # Load model checkpoint
        model_path = checkpoint_path
        opt_path = os.path.join(logger.get_dir(), "checkpoints", f"opt{step:06d}.pt")
        static_vae_opt_path = os.path.join(logger.get_dir(), "checkpoints", f"static_vae_opt{step:06d}.pt")
        ema_path = os.path.join(logger.get_dir(), "checkpoints", f"ema_deformation_0.9999_{step:06d}.pt")
        ema_static_vae_path = os.path.join(logger.get_dir(), "checkpoints", f"ema_static_vae_0.9999_{step:06d}.pt")
        
        opt_checkpoint = th.load(opt_path, map_location=self.accelerator.device)
        static_vae_opt_checkpoint = th.load(static_vae_opt_path, map_location=self.accelerator.device)
        ema_checkpoint = th.load(ema_path, map_location=self.accelerator.device)
        ema_static_vae_checkpoint = th.load(ema_static_vae_path, map_location=self.accelerator.device)

        # Remove 'module.' prefix from checkpoint keys if present
        model_checkpoint = {k.replace('module.', ''): v for k, v in ema_checkpoint.items()}
        static_vae_checkpoint = {k.replace('module.', ''): v for k, v in ema_static_vae_checkpoint.items()}
        opt_checkpoint = {k.replace('module.', ''): v for k, v in opt_checkpoint.items()}
        
        self.model.load_state_dict(model_checkpoint)
        self.static_vae_backbone.load_state_dict(static_vae_checkpoint)
        self.model_params = list(self.model.parameters())
        self.static_vae_params = list(self.static_vae_backbone.parameters())
        self.master_params = self.model_params
        self.static_vae_master_params = self.static_vae_params
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
        
        # First phase: static VAE only
        if self.step < self.static_vae_steps:
            self.forward_backward_static(batch, model_kwargs)
        # Second phase: joint training
        else:
            self.forward_backward(batch, model_kwargs)
            
        step_time = time.time() - start_time
        logger.logkv_mean("step_time", step_time)
        self.log_step()

    def forward_backward_static(self, batch, model_kwargs):
        """Training step for static VAE only (first 10k iterations)"""
        static_feat = model_kwargs['static_feat']
        canonical_feat = static_feat.to(self.accelerator.device)
        extrinsics, intrinsics, image, alpha = build_static_cam(model_kwargs["static_cams"], self.accelerator.device)
        
        with self.accelerator.accumulate(self.static_vae_backbone):
            static_losses, static_gs_model, aux = self.static_vae.training_losses(canonical_feat, image, extrinsics, intrinsics, return_aux=True)
            
            # save rendered images and gt images
            if self.step % 100 == 0 and self.accelerator.is_main_process:
                s_path = os.path.join(logger.get_dir(), 'static_train_images')
                os.makedirs(s_path,exist_ok=True)
                output_image = aux['rec_image'][0].clamp(0.0, 1.0)
                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "render_iter_{:08}.png".format(self.step)), rgb_map)
                rgb_map = aux['gt_image'][0].squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "gt_iter_{:08}.png".format(self.step)), rgb_map)

            loss = static_losses["loss"]
            
            log_loss_dict(static_losses)
            self.accelerator.backward(loss)
            self.optimize_static()

    def optimize_static(self):
        """Optimization step for static VAE only (first 10k iterations)"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.static_vae_params, 1.0)
        
        self.static_vae_opt.step()
        self.warmup_scheduler.step()
        self.static_vae_opt.zero_grad()
        
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        
        # Update static_vae EMA only
        for rate, params in zip(self.ema_rate, self.ema_params_static_vae):
            update_ema(params, self.static_vae_master_params, rate=rate)

    def forward_backward(self, batch, model_kwargs):
        """Joint training step for both static VAE and deformation model (after 10k iterations)"""
        micro_static_pc = batch[0].to(self.accelerator.device)
        micro_delta_pc = batch[1].to(self.accelerator.device)
        B, T = micro_delta_pc.shape[0], micro_delta_pc.shape[1]
        
        static_feat = model_kwargs['static_feat']
        canonical_feat = static_feat.to(self.accelerator.device)
        extrinsics, intrinsics, image, alpha = build_static_cam(model_kwargs["static_cams"], self.accelerator.device)
        
        with self.accelerator.accumulate(self.model, self.static_vae_backbone):
            static_losses, static_gs_model, aux = self.static_vae.training_losses(canonical_feat, image, extrinsics, intrinsics, return_aux=True)
            
            if self.step % 100 == 0 and self.accelerator.is_main_process:
                s_path = os.path.join(logger.get_dir(), 'static_train_images')
                os.makedirs(s_path,exist_ok=True)
                output_image = aux['rec_image'][0].clamp(0.0, 1.0)
                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "render_iter_{:08}.png".format(self.step)), rgb_map)
                rgb_map = aux['gt_image'][0].squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "gt_iter_{:08}.png".format(self.step)), rgb_map)

            static_gs = [get_gaussian_tensor(static_gs_model['MipGS'][idx]) for idx in range(len(static_gs_model['MipGS']))]
            
            loss = static_losses["loss"]
            losses = static_losses
            
            _, valid_idx = pad_static_gs(static_gs)
            output = self.model(static_gs, micro_static_pc, micro_delta_pc)

            if 'kl' in output:
                kl_loss = output['kl'].mean()
            else:
                kl_loss = th.tensor([0.]).to(self.accelerator.device)
            
            loss = loss + kl_loss * self.kl_weight
            losses["delta_kl"] = kl_loss
            pred_delta = output['logits']

            # Compute interpolation loss
            moving_pc = micro_static_pc.unsqueeze(1).repeat(1, T, 1, 1) + micro_delta_pc
            interpolation_loss, interp_losses, estimated_delta = self.interpolation_loss_func(
                static_gs, micro_static_pc, moving_pc, pred_delta, B, self.knn_k, beta=self.beta
            )

            losses.update(interp_losses)
            loss = loss + interpolation_loss * self.xyz_loss_weight

            pred_imgs, gt_imgs = [], []
            for b in range(B):
                for cam_idx in range(model_kwargs["cams"]["extrinsics"].shape[1]):
                    idx = model_kwargs["cams"]["extrinsics"].shape[1] * b + cam_idx
                    extrinsics = model_kwargs["cams"]["extrinsics"][b][cam_idx].to(self.accelerator.device)
                    intrinsics = model_kwargs["cams"]["intrinsics"][b][cam_idx].to(self.accelerator.device)
                    timestep_idx = model_kwargs["cams"]["timestep_idx"][b][cam_idx].item()
                    pred_delta_b = pred_delta[b][cam_idx, :valid_idx[b]]
                    res = self.static_vae.renderers["MipGS"].render(static_gs_model['MipGS'][b], extrinsics, intrinsics, delta_pc=pred_delta_b, detach_static=False)

                    pred_imgs.append(res["rgb"])
                    gt_imgs.append(model_kwargs["cams"]["image"][b][cam_idx].to(self.accelerator.device))

            pred_img = th.stack(pred_imgs, dim=0)
            gt_img = th.stack(gt_imgs, dim=0)
            pixel_l1_loss = self.L1loss(pred_img, gt_img) * self.render_l1_weight
            vgg_loss = self.vgg(pred_img*2 - 1., gt_img*2 - 1.) * self.render_lpips_weight
            ssim_loss = (1.0 - ssim(pred_img, gt_img)) * self.render_ssim_weight
            losses["deformation_l1_loss"] = th.tensor([pixel_l1_loss])
            losses["deformation_vgg_loss"] = th.tensor([vgg_loss])
            losses["deformation_ssim_loss"] = th.tensor([ssim_loss])
            loss = loss + pixel_l1_loss + vgg_loss + ssim_loss

            if self.step % 100 == 0 and self.accelerator.is_main_process:
                s_path = os.path.join(logger.get_dir(), 'train_images')
                os.makedirs(s_path,exist_ok=True)
                output_image = pred_img[-1].clamp(0.0, 1.0)
                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "render_iter_{:08}_t_{:02}.png".format(self.step, int(timestep_idx))), rgb_map)

                rgb_map = gt_img[-1].squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "gt_iter_{:08}_t_{:02}.png".format(self.step, int(timestep_idx))), rgb_map)
            
            if self.use_tensorboard and self.step % (self.log_interval*10) == 0 and self.accelerator.is_main_process:
                self.writer.write_dict({k: losses[k].item() for k in losses}, self.step)
            
            log_loss_dict(losses)
            self.accelerator.backward(loss)
            self.optimize()

    def optimize(self):
        """Joint optimization step for both static VAE and deformation model (after 10k iterations)"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model_params + self.static_vae_params, 1.0)
        
        self.opt.step()
        self.static_vae_opt.step()
        self.warmup_scheduler.step()
        self.opt.zero_grad()
        self.static_vae_opt.zero_grad()
        
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        # Get scale after optimizer step
        current_scale = self.accelerator.scaler.get_scale()
        logger.logkv("lg_grad_scale", np.log(current_scale))
        
        # Update EMA for both models
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        for rate, params in zip(self.ema_rate, self.ema_params_static_vae):
            update_ema(params, self.static_vae_master_params, rate=rate)

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        if not self.accelerator.is_main_process:
            return

        def save_checkpoint(rate, params, name="deformation"):
            state_dict = self._master_params_to_state_dict(params, name)
            logger.log(f"saving model {name} {rate}...")
            if not rate:
                filename = f"{name}_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{name}_{rate}_{(self.step+self.resume_step):06d}.pt"
            with open(os.path.join(get_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.master_params, name="deformation")
        save_checkpoint(0, self.static_vae_master_params, name="static_vae")
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params, name="deformation")
        for rate, params in zip(self.ema_rate, self.ema_params_static_vae):
            save_checkpoint(rate, params, name="static_vae")

        with open(
            os.path.join(get_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        with open(
            os.path.join(get_logdir(), f"static_vae_opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.static_vae_opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params, name="deformation"):
        if name == "deformation":
            state_dict = self.model.state_dict()
            for i, (name, _value) in enumerate(self.model.named_parameters()):
                assert name in state_dict
                state_dict[name] = master_params[i]
        elif name == "static_vae":
            state_dict = self.static_vae_backbone.state_dict()
            param_idx = 0
            # Only update parameters that require gradients
            for name, _value in self.static_vae_backbone.named_parameters():
                if _value.requires_grad:
                    assert name in state_dict
                    state_dict[name] = master_params[param_idx]
                    param_idx += 1
        else:
            raise ValueError(f"Invalid model name: {name}")
        return state_dict

    def _state_dict_to_master_params(self, state_dict, name="deformation"):
        if name == "deformation":
            params = [state_dict[name] for name, _ in self.model.named_parameters()]
        elif name == "static_vae":
            params = [state_dict[name] for name, _ in self.static_vae_backbone.named_parameters()]
        else:
            raise ValueError(f"Invalid model name: {name}")
        return params


def get_logdir():
    p = os.path.join(logger.get_dir(), "checkpoints")
    os.makedirs(p,exist_ok=True)
    return p


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())


def build_static_cam(cam_dict, device):
    '''
    Given cam_dict, stack the keys of extrinsics and intrinsics
    return extrinsics of shape (N, 4, 4) and intrinsics of shape (N, 3, 3)
    '''
    extrinsics = cam_dict["extrinsics"].squeeze(1).to(device)
    intrinsics = cam_dict["intrinsics"].squeeze(1).to(device)
    image = cam_dict["image"].squeeze(1).to(device)
    alpha = cam_dict["alpha"].squeeze(1).to(device)
    return extrinsics, intrinsics, image, alpha


def get_gaussian_tensor(gaussians):
    xyz = gaussians.get_xyz
    features = gaussians.get_features
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    return th.cat([xyz, features.squeeze(), opacity, scales, rotations], dim=-1)


def pad_static_gs(static_gs):
    # pad static_gs to the same length as the longest static_gs, and return mask
    max_len = max([static_gs[i].shape[0] for i in range(len(static_gs))])
    padding = th.zeros((1, static_gs[0].shape[1])).to(static_gs[0].device)
    padding[0, 10] = 1.0  # Set the 11th element (index 10) to 1
    padded_static_gs = [th.cat([static_gs[i], padding.repeat(max_len - static_gs[i].shape[0], 1)], dim=0) for i in range(len(static_gs))]
    padded_static_gs = th.stack(padded_static_gs, dim=0)
    idx = [static_gs[i].shape[0] for i in range(len(static_gs))]
    return padded_static_gs, idx


def compute_interpolation_loss_delta_interp(static_gs, micro_static_pc, micro_moving_pc, output, B, knn_k=4, adaptive_radius=True, beta=7.0):
    """
    Compute interpolation loss between predicted and KNN-estimated deltas
    Args:
        static_gs: List of static gaussian tensors
        micro_static_pc: Static point cloud [B, N_points, 3]
        micro_moving_pc: Moving point cloud [B, T, N_points, 3]
        output: Model output tensor containing predicted deltas [B, T, N_points, 3]
        B: Batch size
        knn_k: Number of nearest neighbors for interpolation
        adaptive_radius: If True, use adaptive radius-based filtering
    Returns:
        interpolation_loss: Scalar loss tensor
        losses_dict: Dictionary containing the interpolation loss
    """
    # Get xyz from static gaussians for all batches
    gs_xyz_list = [static_gs[b][:, :3] for b in range(B)]
    T = micro_moving_pc.shape[1]
    
    # Handle sampling for each batch while maintaining batch dimension
    sampled_gs_xyz = []
    sampling_indices = []
    with th.no_grad():
        for xyz in gs_xyz_list:
            sampled_gs_xyz.append(xyz)
            sampling_indices.append(None)
        
        # Pad all samples to same length for batched operation
        max_samples = max(xyz.shape[0] for xyz in sampled_gs_xyz)
        padded_gs_xyz = th.stack([
            F.pad(xyz, (0, 0, 0, max_samples - xyz.shape[0])) 
            for xyz in sampled_gs_xyz
        ])  # [B, max_samples, 3]

        # Store original lengths for proper masking
        gs_lengths = th.tensor([len(sampled_gs_xyz[b]) for b in range(B)]).to(th.long).to(micro_static_pc.device)
        padding_mask = th.arange(max_samples).expand(B, -1).to(micro_static_pc.device) < gs_lengths.unsqueeze(-1)  # [B, max_samples]

        # Find K nearest neighbors and distances
        knn_dists, knn_idx, _ = pytorch3d.ops.knn_points(
            padded_gs_xyz,
            micro_static_pc,
            lengths1=gs_lengths,
            K=knn_k
        )

        adaptive_radii = knn_dists.mean(dim=-1).sqrt() + 1e-6  # [B, max_samples]
        # Apply adaptive radius-based filtering
        if adaptive_radius:
            radius_mask = knn_dists <= adaptive_radii[..., None] ** 2
            weights = th.exp(-beta * knn_dists / adaptive_radii[..., None] ** 2)
            weights = weights * radius_mask.float()
        else:
            weights = th.exp(-beta * knn_dists)

        # Apply padding mask to weights
        weights = weights * padding_mask.unsqueeze(-1)

        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather and weight point movements
        batch_idx = th.arange(B).view(-1, 1, 1, 1).expand(-1, max_samples, knn_k, T).to(micro_static_pc.device)
        time_idx = th.arange(T).view(1, 1, 1, -1).expand(B, max_samples, knn_k, -1).to(micro_static_pc.device)
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, T)
        
        # Get movements of neighboring points
        neighbor_positions = micro_moving_pc[batch_idx, time_idx, knn_idx_expanded]  # [B, max_samples, K, T, 3]
        neighbor_movements = neighbor_positions - micro_static_pc[batch_idx, knn_idx_expanded]
        
        # Apply weights to compute interpolated movement
        weights = weights.unsqueeze(-1).unsqueeze(-2)  # [B, max_samples, K, 1, 1]
        interpolated_movements = (weights * neighbor_movements).sum(dim=2)  # [B, max_samples, T, 3]
        estimated_deltas = interpolated_movements.transpose(1, 2)  # [B, T, max_samples, 3]

    # Get predicted deltas for all batches
    pred_deltas = []
    for b in range(B):
        if sampling_indices[b] is not None:
            pred_deltas.append(output[b, :, sampling_indices[b], :3])
        else:
            pred_deltas.append(output[b, :, :sampled_gs_xyz[b].shape[0], :3])
    
    # Compute loss for all batches at once
    valid_samples = th.tensor([xyz.shape[0] for xyz in sampled_gs_xyz]).to(estimated_deltas.device)
    mask = th.arange(max_samples).to(estimated_deltas.device).expand(B, -1) < valid_samples.unsqueeze(1)
    mask = mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, max_samples]

    pred_deltas = th.stack([
        F.pad(delta, (0, 0, 0, max_samples - delta.shape[1])) 
        for delta in pred_deltas
    ])  # [B, T, max_samples, 3]

    # Compute masked L1 loss
    diff = th.abs((pred_deltas - estimated_deltas))  # [B, T, max_samples, 3]
    masked_diff = diff * mask.unsqueeze(-1)  # Apply mask
    interpolation_loss = masked_diff.sum() / (mask.sum() * 3)  # Normalize by number of valid points and xyz dimensions

    losses_dict = {"deformation_xyz_loss": th.tensor([interpolation_loss])}
    
    return interpolation_loss, losses_dict, estimated_deltas
