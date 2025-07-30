import os
import random
import argparse
import numpy as np
import time
from collections import OrderedDict
from tqdm import tqdm
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from omegaconf import OmegaConf
from accelerate import Accelerator
import imageio
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from dataset.dataset_latent_inference import load_data
from model.dit import DiT
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from model.autoencoder import GSKLTemporalVariationalAutoEncoder
from model.sparse_voxel_diffusion.sparse_transformer_vae import SparseTransformerVAE
from model.sparse_voxel_diffusion.sparse_vae import SparseVAE
from utils import logger
from utils.script_util import create_gaussian_diffusion
from utils.inference_utils import align_gaussian_to_canonical, sample_gs, render_and_save_images, seed_everything, download_model_files, download_example_assets

from train_vae import pad_static_gs, get_gaussian_tensor
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

MODEL_TYPES = {
    'xstart': 'x_start',
    'v': 'v',
    'eps': 'noise',
}


def main():
    args = create_argparser().parse_args()

    # If model_name is provided, download all necessary files
    if args.model_name is not None:
        print(f"Downloading {args.model_name} files from Hugging Face...")
        downloaded_files = download_model_files(args.model_name)
        
        # Override arguments with downloaded files
        args.ckpt = downloaded_files["ckpt"]
        args.vae_ckpt = downloaded_files["vae_ckpt"]
        args.static_vae_ckpt = downloaded_files["static_vae_ckpt"]
        args.static_mean_file = downloaded_files["static_mean"]
        args.static_std_file = downloaded_files["static_std"]
        args.deformation_mean_file = downloaded_files["deformation_mean"]
        args.deformation_std_file = downloaded_files["deformation_std"]
        
        print("Model files downloaded successfully.")
    
    # Download example assets if requested
    if args.download_assets:
        download_example_assets(args.model_name, args.data_dir)

    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large", static_vae_ckpt=args.static_vae_ckpt)
    pipeline.cuda()

    model_and_diffusion_config = OmegaConf.load(args.config)
    print("Model and Diffusion config: ", model_and_diffusion_config)

    seed_everything(args.seed)

    # Initialize models
    model = DiT(**model_and_diffusion_config['model'])
    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])

    # Load model checkpoint
    if args.ckpt is not None:
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.ckpt, map_location="cpu")
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("Loaded ckpt: ", args.ckpt)

    # Initialize VAE models
    model_and_diffusion_config['motion_vae']['num_timesteps'] = args.num_timesteps
    vae = GSKLTemporalVariationalAutoEncoder(**model_and_diffusion_config['motion_vae'])
    static_vae_model = SparseTransformerVAE(**model_and_diffusion_config['static_vae']['backbones']['vae']['args'])
    backbone = {'vae': static_vae_model}
    static_vae = SparseVAE(backbone, **model_and_diffusion_config['static_vae']['framework']['args'])
    backbone = list(backbone.values())[0]

    # Load VAE checkpoints
    if args.vae_ckpt is not None:
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.vae_ckpt, map_location="cpu")
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        vae.load_state_dict(new_state_dict)
        print("Loading Motion VAE from: ", args.vae_ckpt)
    
    if args.static_vae_ckpt is not None:
        state_dict = torch.load(args.static_vae_ckpt, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        static_vae_model.load_state_dict(new_state_dict, strict=False)
        print("Loading Static VAE from: ", args.static_vae_ckpt)
    
    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision='fp16' if args.use_fp16 else 'no',
        project_dir=logger.get_dir(),
    )
    if accelerator.is_main_process:
        logger.save_args(options)

    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        in_the_wild=args.in_the_wild,
        deterministic=True,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        load_camera=args.load_camera,
        num_timesteps=args.num_timesteps,
    )

    # Prepare models and data
    model, vae, static_vae, backbone, val_data = accelerator.prepare(
        model, vae, static_vae, backbone, val_data
    )

    model.eval()
    vae.eval()
    backbone.eval()

    static_mean = torch.load(args.static_mean_file).to(torch.float32).to(accelerator.device) if args.static_mean_file is not None else 0
    static_std = torch.load(args.static_std_file).to(torch.float32).to(accelerator.device) if args.static_std_file is not None else 1
    deformation_mean = torch.load(args.deformation_mean_file).to(torch.float32).to(accelerator.device) if args.deformation_mean_file is not None else 0
    deformation_std = torch.load(args.deformation_std_file).to(torch.float32).to(accelerator.device) if args.deformation_std_file is not None else 1

    # Setup noise schedule
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(accelerator.device))

    img_id = 0
    num_batch_per_rank = args.num_samples // accelerator.num_processes // args.batch_size
    
    for k, v in static_vae.renderers.items():
        v.rendering_options.resolution = 512
    
    if os.path.exists(os.path.join(logger.get_dir(), 'align_gaussian_images')):
        import shutil
        shutil.rmtree(os.path.join(logger.get_dir(), 'align_gaussian_images'))

    for _ in tqdm(range(num_batch_per_rank)):
        model_kwargs = next(val_data)

        with torch.no_grad():
            static_gs = []
            static_gs_model = []
            scale_factors = []
            for b in range(args.batch_size):
                image = Image.open(model_kwargs['canonical_path'][b]).convert("RGBA")
                outputs = pipeline.run(
                        image,
                        seed=args.seed,
                        sparse_structure_sampler_params={
                            "steps": 12,
                            "cfg_strength": 7.5,
                        },
                        slat_sampler_params={
                            "steps": 12,
                            "cfg_strength": 3,
                        }
                    )
                
                if 'mask_path' in model_kwargs:
                    im_data = np.array(image.convert("RGB")).astype(np.float32)
                    bg = np.array([1,1,1]).astype(np.float32)
                    norm_data = im_data / 255.0
                    alpha = np.array(Image.open(model_kwargs['mask_path'][b]).convert("L")).astype(np.float32) / 255.0
                else:
                    im_data = np.array(image.convert("RGBA")).astype(np.float32)
                    bg = np.array([1,1,1]).astype(np.float32)
                    norm_data = im_data / 255.0
                    alpha = norm_data[:, :, 3:4]
                    norm_data = norm_data[:,:,:3] * alpha + bg * (1 - alpha)
                
                canonical_image = torch.from_numpy(norm_data).permute(2, 0, 1).to(accelerator.device)
                aligned_static_gs_model, scale_factor = align_gaussian_to_canonical(outputs['gaussian'][0], canonical_image, torch.from_numpy(alpha).squeeze().to(accelerator.device), model_kwargs["cams"]["intrinsics"][b][0], static_vae, id=img_id+b, device=accelerator.device, in_the_wild=args.in_the_wild)
                static_gs_model.append(aligned_static_gs_model)
                static_gs.append(get_gaussian_tensor(aligned_static_gs_model))
                scale_factors.append(scale_factor)
            
            fps_static_gs = sample_gs(static_gs, num_latents=model_and_diffusion_config['motion_vae']['num_latents'], device=accelerator.device)
            fps_static_gs_4096 = sample_gs(static_gs, num_latents=4096, device=accelerator.device)
            padded_static_gs, valid_idx = pad_static_gs(static_gs)

            # Setup conditions for diffusion model
            condition = {
                'cond_images': model_kwargs['cond_images'].to(accelerator.device),
                'static_latent': (fps_static_gs_4096 - static_mean) / static_std,
                'deformation_position_xyz': fps_static_gs[..., :3]
            }
            unconditional_condition = {
                'cond_images': torch.zeros_like(model_kwargs['cond_images']).to(accelerator.device),
                'static_latent': (fps_static_gs_4096 - static_mean) / static_std,
                'deformation_position_xyz': fps_static_gs[..., :3]
            }

            # Setup model wrapper for sampling
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type=MODEL_TYPES[model_and_diffusion_config['diffusion']['predict_type']],
                model_kwargs={},
                guidance_type='classifier-free',
                guidance_scale=args.guidance_scale,
                guidance_scale2=args.guidance_scale2,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

            sample_shape = (args.batch_size, args.num_timesteps, model_and_diffusion_config['model']['resolution'], model_and_diffusion_config['model']['in_channels'])
            # Sample from diffusion model
            noise = torch.randn(sample_shape, device=accelerator.device)
            samples = dpm_solver.sample(
                x=noise,
                steps=args.rescale_timesteps,
                t_start=1.0,
                t_end=1/1000,
                order=2,
                skip_type='time_uniform',
                method='adaptive' if args.adaptive else 'multistep',
            )
            samples_denorm = samples * deformation_std + deformation_mean
           
            B, T, N, C = samples_denorm.shape
            samples_denorm = samples_denorm.reshape(B*T, N, C)

            # Decode samples through VAE
            with accelerator.autocast():
                pred_delta = vae.decode(samples_denorm, padded_static_gs)

            pred_delta = pred_delta.to(torch.float32)
            # Render and save results
            render_and_save_images(
                args=args,
                static_vae=static_vae,
                static_gs_model=static_gs_model,
                pred_delta=pred_delta,
                model_kwargs=model_kwargs,
                valid_idx=valid_idx,
                img_id=img_id,
                accelerator=accelerator,
                scale_factors=scale_factors,
                save_dir='inference_images'
            )
        img_id += args.batch_size


def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--model_name", type=str, default="GVFDiffusion_v1.0",
                       help="Name of the model to download from Hugging Face Hub")
    parser.add_argument("--download_assets", action="store_true", 
                       help="Download example assets from Hugging Face Hub")
    parser.add_argument("--assets_dir", type=str, default="./assets",
                       help="Directory to store downloaded assets")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/diffusion.yml")
    # Data args
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--txt_file", type=str, default="video_names.txt")
    parser.add_argument("--static_mean_file", type=none_or_str, default=None)
    parser.add_argument("--static_std_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_mean_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_std_file", type=none_or_str, default=None)
    parser.add_argument("--load_camera", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=24)
    # Inference args
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--rescale_timesteps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale2", type=float, default=1.0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--static_vae_ckpt", type=str, default=None)
    parser.add_argument("--in_the_wild", action="store_true")
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
