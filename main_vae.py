import os
import argparse
import torch
import torch.utils.cpp_extension
from omegaconf import OmegaConf
from collections import OrderedDict

from utils import logger
from dataset.dataset_vae import load_data
from model.autoencoder import GSKLTemporalVariationalAutoEncoder
from model.sparse_voxel_diffusion.sparse_transformer_vae import SparseTransformerVAE
from model.sparse_voxel_diffusion.sparse_vae import SparseVAE

def main():
    args = create_argparser().parse_args()

    model_and_diffusion_config = OmegaConf.load(args.config)

    model_and_diffusion_config['model']['num_timesteps'] = args.sample_timesteps
    model_and_diffusion_config['model']['knn_k'] = args.knn_k
    model_and_diffusion_config['model']['beta'] = args.beta
    model = GSKLTemporalVariationalAutoEncoder(**model_and_diffusion_config['model'])

    static_vae_model = SparseTransformerVAE(**model_and_diffusion_config['backbones']['vae']['args'])
    backbone = {}
    backbone['vae'] = static_vae_model
    static_vae = SparseVAE(backbone, **model_and_diffusion_config['framework']['args'])
    backbone = list(backbone.values())[0]
    
    # load static vae weight
    if args.static_vae_ckpt is not None:
        state_dict = torch.load(args.static_vae_ckpt, map_location='cpu')
        # Only pop layers with size mismatch
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
        print("Loading static vae weight: ", msg)
        if not args.finetune_encoder:
            backbone.freeze_encoder()

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
        print("Loading deformation vae weight: ", msg)
        has_pretrain_weight = True
    
    logger.configure(args.exp_name)
    logger.log("Model and Diffusion config: ", model_and_diffusion_config)
    logger.log("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    logger.log("num of static vae params: {} M".format(sum(p.numel() for p in static_vae_model.parameters())/1e6))
    logger.log("creating data loader...")


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=model_and_diffusion_config['backbones']['vae']['args']['resolution'],
        train=True,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        load_camera=args.load_camera,
        cam_root_path=args.cam_root_path,
        num_pts=model_and_diffusion_config['model']['num_inputs'],
        sample_timesteps=args.sample_timesteps,
    )

    from train_vae import TrainLoop

    logger.log("training...")
    TrainLoop(
        model,
        static_vae,
        static_vae_backbone=backbone,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        use_tensorboard=args.use_tensorboard,
        render_l1_weight=args.render_l1_weight,
        render_lpips_weight=args.render_lpips_weight,
        render_ssim_weight=args.render_ssim_weight,
        has_pretrain_weight=has_pretrain_weight,
        knn_k=args.knn_k,
        beta=args.beta,
        xyz_loss_weight=args.xyz_loss_weight,
        kl_weight=args.kl_weight,
        static_vae_steps=args.static_vae_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        args=args,
        auto_resume=args.auto_resume,
    ).run_loop()

 
def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--static_vae_ckpt", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/vae.yml")
    # Train args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use_vgg", action="store_true")
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--render_l1_weight", type=float, default=1.0)
    parser.add_argument("--render_lpips_weight", type=float, default=0.2)
    parser.add_argument("--render_ssim_weight", type=float, default=0.2)
    parser.add_argument("--kl_weight", type=float, default=1e-5)
    parser.add_argument("--static_vae_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--beta", type=float, default=7.0)
    parser.add_argument("--xyz_loss_weight", type=float, default=0.1)
    parser.add_argument("--sample_timesteps", type=int, default=1)
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--auto_resume", action="store_true")
    # Data args
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="")
    parser.add_argument("--img_feature_root", type=str, default="")
    parser.add_argument("--static_mean_file", type=none_or_str, default=None)
    parser.add_argument("--static_std_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_mean_file", type=none_or_str, default=None)
    parser.add_argument("--deformation_std_file", type=none_or_str, default=None)
    parser.add_argument("--load_camera", type=int, default=0)
    parser.add_argument("--cam_root_path", type=str, default="")
    parser.add_argument("--render_resolution", type=int, default=512)
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
