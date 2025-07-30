import os
import argparse
import torch
import torch.utils.cpp_extension
from omegaconf import OmegaConf
from collections import OrderedDict

from utils import logger
from utils.script_util import create_gaussian_diffusion
from dataset.dataset_latent import load_data
from train_latent import TrainLoop
from model.dit import DiT
from model.resample import UniformSampler

def main():
    args = create_argparser().parse_args()

    model_and_diffusion_config = OmegaConf.load(args.config)

    model = DiT(**model_and_diffusion_config['model'])
    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])

    logger.configure(args.exp_name)
    has_pretrain_weight = False
    if args.ckpt is not None:
        print("Loading pretrain weight from: ", args.ckpt)
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.ckpt, map_location="cpu")
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.log("load pretrain weight msg: ", msg)
        has_pretrain_weight = True

    schedule_sampler = UniformSampler(model_and_diffusion_config['diffusion']['steps'])
    logger.log("Model and Diffusion config: ", model_and_diffusion_config)
    logger.log("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=32,
        train=True,
        uncond_p=args.uncond_p,
        canonical_file=args.canonical_file,
        static_mean_file=args.static_mean_file,
        static_std_file=args.static_std_file,
        deformation_mean_file=args.deformation_mean_file,
        deformation_std_file=args.deformation_std_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
    )

    logger.log("training...")
    TrainLoop(
        model,
        diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        schedule_sampler=schedule_sampler,
        use_tensorboard=args.use_tensorboard,
        has_pretrain_weight=has_pretrain_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mem_ratio=args.mem_ratio,
        args=args,
    ).run_loop()

 
def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--static_vae_ckpt", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/diffusion.yml")
    # Train args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--uncond_p", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="l1")
    parser.add_argument("--static_vae_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--xyz_points", type=int, default=4096)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--sample_timesteps", type=int, default=1)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--mem_ratio", type=float, default=1.0)
    # Data args
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--canonical_file", type=none_or_str, default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="")
    parser.add_argument("--img_feature_root", type=str, default="")
    parser.add_argument("--static_mean_file", type=none_or_str, default=None)
    parser.add_argument("--static_std_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_mean_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_std_file", type=none_or_str, default=None)
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.utils.cpp_extension.CUDA_HOME = "/usr/local/cuda/"
    print("cuda home: ", torch.utils.cpp_extension.CUDA_HOME)
    main()
