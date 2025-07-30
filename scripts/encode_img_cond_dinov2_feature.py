import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_sample(opt, file_path, dinov2_model, transform, rank):
    output_path = os.path.join(opt.output_dir, f'dinov2_cond_features_{opt.timesteps}', f'{file_path}.npz')
    if os.path.exists(output_path):
        try:
            np.load(output_path, allow_pickle=True) 
            print(f"Skipping {file_path} - features already exist and are valid")
            return
        except:
            print(f"Found corrupted feature file for {file_path} - regenerating")

    # Initialize lists to store features for all timesteps
    all_patchtokens = []
    all_clstokens = []

    # Process all 24 timesteps
    for timestep in range(opt.timesteps):
        # Load frontal image
        image_path = os.path.join(opt.input_dir, file_path, 'imgs', f'timestep_{timestep:02d}_view_00.png')
        image = Image.open(image_path)
        image = image.resize((518, 518), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255
        image = image[:, :, :3] * image[:, :, 3:]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Extract features
        with torch.no_grad():
            features = dinov2_model(transform(image).unsqueeze(0).cuda(rank), is_training=True)

        # Store features
        if opt.normalize:
            all_patchtokens.append(features['x_norm_patchtokens'].squeeze(0).cpu().numpy())
            all_clstokens.append(features['x_norm_clstoken'].squeeze(0).cpu().numpy())
        else:
            all_patchtokens.append(features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].squeeze(0).cpu().numpy())
            all_clstokens.append(features['x_prenorm'][:, 0].squeeze(0).cpu().numpy())

    # Concatenate features from all timesteps
    pack = {
        'patchtokens': np.stack(all_patchtokens, axis=0).astype(np.float16),  # Shape: [24, num_patches, feature_dim]
        'clstoken': np.stack(all_clstokens, axis=0).astype(np.float16),  # Shape: [24, feature_dim]
        'normalized': opt.normalize
    }
    features = np.concatenate([pack['patchtokens'], pack['clstoken'][:, None, :]], axis=1)

    os.makedirs(os.path.join(opt.output_dir, f'dinov2_cond_features_{opt.timesteps}'), exist_ok=True)
    np.savez_compressed(output_path, features=features)

def main(rank, world_size, opt):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    os.makedirs(opt.output_dir, exist_ok=True)

    # Only rank 0 downloads the model first
    if rank == 0:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    # Synchronize all processes
    dist.barrier()
    # Now other ranks can safely load the model
    if rank != 0:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    
    dinov2_model.eval().cuda(rank)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Distribute files across GPUs
    with open(opt.txt_file, 'r') as f:
        file_list = f.read().splitlines()[opt.start_idx:opt.end_idx]
    
    start = len(file_list) * rank // world_size
    end = len(file_list) * (rank + 1) // world_size
    file_list = file_list[start:end]

    # Extract features
    for file_path in tqdm(file_list):
        try:
            process_sample(opt, file_path, dinov2_model, transform, rank)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/objaverse_4d_rendering/',
                        help='Input directory containing OBJ files in subdirectories')
    parser.add_argument('--output_dir', type=str, default='./output/objaverse_4d_dinov2_features',
                        help='Directory to save the features')
    parser.add_argument('--txt_file', type=str, default='./assets/4d_objs.txt',
                        help='Text file containing list of files to process')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=10, help='End index')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize features')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--timesteps', type=int, default=24,
                        help='Number of timesteps to process')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.gpus > 1:
        mp.spawn(
            main,
            args=(opt.gpus, opt),
            nprocs=opt.gpus,
            join=True
        )
    else:
        main(0, 1, opt)

