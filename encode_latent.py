import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from collections import OrderedDict
from accelerate import Accelerator

from train_vae import build_static_cam, pad_static_gs, get_gaussian_tensor
from model.sparse_voxel_diffusion.sparse_transformer_vae import SparseTransformerVAE
from model.sparse_voxel_diffusion.sparse_vae import SparseVAE
from utils import logger
from utils.script_util import psnr
import imageio
from tqdm import tqdm
from torch_cluster import fps

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_static_images(aux, save_dir, img_id):
    s_path = os.path.join(logger.get_dir(), save_dir)
    os.makedirs(s_path, exist_ok=True)
    B = aux['rec_image'].shape[0]
    val_psnrs = []
    for b in range(B):
        output_image = aux['rec_image'][b].clamp(0.0, 1.0)
        rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
        rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
        imageio.imwrite(os.path.join(s_path, f"render_{img_id+b:06d}.png"), rgb_map)

        gt_image = aux['gt_image'][b].clamp(0.0, 1.0)
        rgb_map = gt_image.squeeze().permute(1, 2, 0).cpu() 
        rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
        imageio.imwrite(os.path.join(s_path, f"gt_{img_id+b:06d}.png"), rgb_map)
        val_psnrs.append(psnr(output_image, gt_image).cpu().numpy())
    return val_psnrs


def render_and_save_images(
    args,
    static_vae, 
    static_gs_model, 
    pred_delta, 
    model_kwargs, 
    valid_idx, 
    img_id, 
    accelerator, 
    save_dir='val_images'
):
    """
    Render and save images for both predictions and ground truth
    
    Args:
        static_vae: Static VAE model
        static_gs_model: Static Gaussian model
        pred_delta: Predicted delta values
        model_kwargs: Dictionary containing camera parameters and images
        valid_idx: Valid indices for the predictions
        img_id: Current image ID
        accelerator: Accelerator instance
        save_dir: Directory name for saving rendered images
    """
    B = pred_delta.shape[0]
    val_psnrs = []
    
    for b in range(B):
        for cam_idx in range(model_kwargs["cams"]["extrinsics"].shape[1]):
            idx = model_kwargs["cams"]["extrinsics"].shape[1] * b + cam_idx
            extrinsics = model_kwargs["cams"]["extrinsics"][b][cam_idx].to(accelerator.device)
            intrinsics = model_kwargs["cams"]["intrinsics"][b][cam_idx].to(accelerator.device)
            timestep_idx = model_kwargs["cams"]["timestep_idx"][b][cam_idx].item()
            pred_delta_b = pred_delta[b][timestep_idx, :valid_idx[b]]

            for k, v in static_vae.renderers.items():
                v.rendering_options.resolution = 512
            
            res = static_vae.renderers["MipGS"].render(
                static_gs_model['MipGS'][b], 
                extrinsics, 
                intrinsics, 
                delta_pc=pred_delta_b.to(torch.float32)
            )

            # Save rendered images
            s_path = os.path.join(logger.get_dir(), save_dir)
            os.makedirs(s_path, exist_ok=True)
            
            output_image = res["rgb"].clamp(0.0, 1.0)
            gt_image = model_kwargs["cams"]["image"][b][cam_idx].to(accelerator.device)
            val_psnrs.append(psnr(output_image, gt_image).cpu().numpy())
            
            rgb_map = output_image.squeeze().permute(1, 2, 0).cpu()
            rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
            imageio.imwrite(
                os.path.join(s_path, f"render_{img_id+b:06d}_cam_{cam_idx%args.load_camera:03d}_timesteps_{timestep_idx:02d}.png"),
                rgb_map
            )

            gt_s_path = os.path.join(logger.get_dir(), 'gt_images')
            os.makedirs(gt_s_path, exist_ok=True)
            rgb_map = gt_image.squeeze().permute(1, 2, 0).cpu() 
            rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
            imageio.imwrite(
                os.path.join(gt_s_path, f"gt_{img_id+b:06d}_cam_{cam_idx%args.load_camera:03d}_timesteps_{timestep_idx:02d}.png"), 
                rgb_map
            )
    
    return val_psnrs


def sample_gs(static_gs_list, num_latents, device):
    gs_lengths = [gs.shape[0] for gs in static_gs_list]
    B = len(gs_lengths)
        
    # Create batch indices for gaussians
    gs_batch = []
    for i in range(len(gs_lengths)):
        gs_batch.extend([i] * gs_lengths[i])
    gs_batch = torch.tensor(gs_batch, device=device)
    
    # Stack all gaussians into one tensor
    stacked_gs = torch.cat([static_gs_list[i][:, :3] for i in range(len(static_gs_list))], dim=0)
    
    # Calculate ratios for each instance based on number of gaussians
    gs_ratios = torch.tensor([(num_latents / gs_len) for gs_len in gs_lengths], device=device)
    gs_idx = fps(stacked_gs, gs_batch, ratio=gs_ratios)
    input_static_gs = stacked_gs[gs_idx].reshape(B, num_latents, 3)
    sampled_static_gs = torch.cat([static_gs_list[i] for i in range(len(static_gs_list))], dim=0)[gs_idx].reshape(B, num_latents, 14)
    return sampled_static_gs


def main():
    args = create_argparser().parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='fp16' if args.use_fp16 else 'no',
    )

    logger.configure(args.exp_name)
    
    # Load config and set seed
    model_and_diffusion_config = OmegaConf.load(args.config)
    if accelerator.is_main_process:
        print("Model and Diffusion config: ", model_and_diffusion_config)
    seed_everything(args.seed + accelerator.process_index)

    # Initialize model as GSKLTemporalVariationalAutoEncoder
    from model.autoencoder import GSKLTemporalVariationalAutoEncoder
    model = GSKLTemporalVariationalAutoEncoder(**model_and_diffusion_config['model'])
    static_vae_model = SparseTransformerVAE(**model_and_diffusion_config['backbones']['vae']['args'])
    backbone = {'vae': static_vae_model}
    static_vae = SparseVAE(backbone, **model_and_diffusion_config['framework']['args'])
    backbone = list(backbone.values())[0]

    # Load checkpoints
    if args.static_vae_ckpt is not None:
        state_dict = torch.load(args.static_vae_ckpt, map_location='cpu')
        if 'out_layer.weight' in state_dict and state_dict['out_layer.weight'].shape != backbone.out_layer.weight.shape:
            state_dict.pop('out_layer.weight')
            state_dict.pop('out_layer.bias')
            state_dict.pop('MipGS_perturbation')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        msg = backbone.load_state_dict(new_state_dict, strict=False)
        if accelerator.is_main_process:
            print("Static VAE loading message:", msg)

    if args.ckpt is not None:
        if accelerator.is_main_process:
            print("Loading model weights from:", args.ckpt)
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.ckpt, map_location="cpu")
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # Calculate data range for this process
    total_samples = args.end_idx - args.start_idx
    samples_per_process = total_samples // accelerator.num_processes
    process_start_idx = args.start_idx + samples_per_process * accelerator.process_index
    process_end_idx = process_start_idx + samples_per_process if accelerator.process_index < accelerator.num_processes - 1 else args.end_idx

    from dataset.dataset_encode import load_data
    # Initialize data loader
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=model_and_diffusion_config['backbones']['vae']['args']['resolution'],
        train=False,
        deterministic=True,
        start_idx=process_start_idx,
        end_idx=process_end_idx,
        txt_file=args.txt_file,
        load_camera=args.load_camera,
        cam_root_path=args.cam_root_path,
        num_pts=model_and_diffusion_config['model']['num_inputs'],
        sample_timesteps=args.timesteps,
    )

    # Prepare models and data
    model, static_vae, backbone, val_data = accelerator.prepare(
        model, static_vae, backbone, val_data
    )

    model.eval()
    backbone.eval()

    # Create output directory for latents
    latents_dir = os.path.join(logger.get_dir(), 'latents')
    if accelerator.is_main_process:
        os.makedirs(latents_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Calculate batches for this process
    samples_per_process = args.num_samples // accelerator.num_processes
    num_batch_per_process = samples_per_process // args.batch_size
    
    progress_bar = tqdm(
        range(num_batch_per_process),
        disable=not accelerator.is_main_process,
        desc=f"Process {accelerator.process_index} encoding"
    )
    
    val_deformation_psnrs = []
    val_static_psnrs = []
    img_id = 0
    for _ in progress_bar:
        batch, model_kwargs = next(val_data)

        with torch.no_grad():
            # Process static features
            static_feat = model_kwargs['static_feat']
            canonical_feat = static_feat.to(accelerator.device)

            basename = os.path.basename(model_kwargs['path'][0])
            save_path = os.path.join(latents_dir, basename, "deformation_latent.pt")
            
            if os.path.exists(save_path) and not args.num_augmentations > 1:
                print(f"Skipping {save_path} - deformation latent already exists")
                continue

            if args.debug:
                extrinsics, intrinsics, image, alpha = build_static_cam(model_kwargs["static_cams"], accelerator.device)
                static_gs_model, aux = static_vae.encode_decode(canonical_feat, image, extrinsics, intrinsics, return_aux=True)
                val_static_psnrs.extend(save_static_images(aux, "static_val_images", img_id))

            with accelerator.autocast():
                # Encode static features
                static_gs_model, aux = static_vae.encode_decode_no_render(canonical_feat, return_aux=True)
                micro_static_pc = batch[0].to(accelerator.device)
                micro_delta_pc = batch[1].to(accelerator.device)

                # Encode deformation
                for n_augmentation in range(args.num_augmentations):
                    static_gs = [get_gaussian_tensor(static_gs_model['MipGS'][idx]) for idx in range(len(static_gs_model['MipGS']))]
                    fps_static_gs_4096 = sample_gs(static_gs, 4096, accelerator.device)
                    kl, x, posterior, sampled_static_pc = model.module.encode(micro_static_pc, micro_delta_pc, static_gs)

                    # Process and save deformation latents
                    latent_mean = posterior.mean.cpu().detach()
                    latent_std = posterior.std.cpu().detach()
                    sampled_static_pc = sampled_static_pc.cpu().detach()

                    N, C = latent_mean.shape[-2:]
                    latent_mean = latent_mean.reshape(args.batch_size, args.timesteps, N, C)
                    latent_std = latent_std.reshape(args.batch_size, args.timesteps, N, C)
                    sampled_static_pc = sampled_static_pc.reshape(args.batch_size, N, 14)
                    fps_static_gs_4096 = fps_static_gs_4096.cpu().detach().reshape(args.batch_size, 4096, 14)
                    
                    for i in range(args.batch_size):
                        batch_mask = aux['x'].coords[:, 0] == i
                        batch_coords = aux['x'].coords[batch_mask, 1:].cpu().detach()
                        static_feats = aux['x'].feats[batch_mask.cpu()].cpu().detach()

                        if torch.isnan(latent_mean[i]).any() or torch.isnan(latent_std[i]).any() or torch.isnan(sampled_static_pc[i]).any() or torch.isnan(fps_static_gs_4096[i]).any():
                            print(f"NaN found in latent {i}")
                            continue

                        deform_dict = {
                            'latent_mean': latent_mean[i],
                            'latent_std': latent_std[i],
                            'fps_sampled_gs_1024': sampled_static_pc[i],
                            'fps_sampled_gs_4096': fps_static_gs_4096[i],
                            'static_gs_feats': static_feats,
                            'static_gs_coords': batch_coords,
                        }
                        
                        basename = os.path.basename(model_kwargs['path'][i])

                        save_path = os.path.join(latents_dir, basename, "deformation_latent.pt")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(deform_dict, save_path)
                    
                        if args.debug:
                            padded_static_gs, valid_idx = pad_static_gs(static_gs)
                            pred_delta = model.decode(x, padded_static_gs)
            if args.debug:
                val_deformation_psnrs.extend(
                    render_and_save_images(
                        args=args,
                        static_vae=static_vae,
                        static_gs_model=static_gs_model,
                        pred_delta=pred_delta,
                        model_kwargs=model_kwargs,
                        valid_idx=valid_idx,
                        img_id=img_id,
                        accelerator=accelerator,
                        save_dir='val_images'
                    )
                )
            img_id += args.batch_size
            
    if val_static_psnrs:
        print("static psnr: ", np.mean(val_static_psnrs))
    if val_deformation_psnrs:
        print("deform psnr: ", np.mean(val_deformation_psnrs))
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All processes completed encoding")

def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--static_vae_ckpt", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/volume_32.yml")
    # Train args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--uncond_p", type=float, default=0.2)
    # Data args
    parser.add_argument("--data_dir", type=str, default="/mnt/blob/output/objaverse_4d_volume/volume_act/")
    parser.add_argument("--canonical_file", type=str, default="/mnt/blob/output/objaverse_4d_deformation_volume_scgs_first_frame_canonical_nodes4096/canonical_frames.csv")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="/mnt/blob/data/4d_volume.txt")
    parser.add_argument("--load_camera", type=int, default=0)
    parser.add_argument("--cam_root_path", type=str, default="/mnt/blob/data/objaverse_4d_rendering_no_light/")
    # Inference args
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--num_augmentations", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
