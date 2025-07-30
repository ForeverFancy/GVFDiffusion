import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
import imageio
import rembg

rembg_session = rembg.new_session('u2net')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def remove_background(image):
    has_alpha = False
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = image
    else:
        image = image.convert('RGB')
        output = rembg.remove(image, session=rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
    
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((380, 380), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    
    paste_position = ((512 - 380) // 2, (512 - 380) // 2)
    # Extract alpha channel and convert to image
    alpha_channel = output[:, :, 3:4]
    new_alpha_channel = np.zeros((512, 512, 1))
    new_alpha_channel[paste_position[1]:paste_position[1]+380, paste_position[0]:paste_position[0]+380] = alpha_channel
    mask = Image.fromarray((new_alpha_channel * 255).astype(np.uint8).squeeze(), mode='L')
    output = output[:, :, :3] * output[:, :, 3:4] + (1 - output[:, :, 3:4]) * 1.0
    
    new_rgb_array = np.ones((512, 512, 3))
    
    # Paste the smaller image into the center of the larger canvas
    new_rgb_array[paste_position[1]:paste_position[1]+380, 
                 paste_position[0]:paste_position[0]+380, :3] = output
    
    output = Image.fromarray((new_rgb_array * 255).astype(np.uint8))
    return output, bbox, mask

def process_sample(opt, sequence_folder, dinov2_model, transform, rank, start_idx=0, sample_rate=1):
    output_path = os.path.join(opt.output_dir, f'dinov2_cond_features', f'{os.path.basename(sequence_folder)}.npz')
    video_path = os.path.join(opt.output_dir, f'video_downsampled', f'{os.path.basename(sequence_folder)}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Get all image files in the sequence folder
    image_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Apply start index and sample rate
    image_files = image_files[start_idx::sample_rate][:32]
    frames = []
    
    # Initialize lists to store features
    all_patchtokens = []
    all_clstokens = []

    bbox = None
    # Process each frame in the sequence
    for image_file in image_files:
        # Load image
        image_path = os.path.join(sequence_folder, image_file)
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)
        if bbox is None:
            image, bbox, mask = remove_background(image)
            os.makedirs(os.path.join(opt.input_dir, "resized_images", f'{os.path.basename(sequence_folder)}'), exist_ok=True)
            mask.save(os.path.join(os.path.join(opt.input_dir, "resized_images", f'{os.path.basename(sequence_folder)}', f'{image_name}_mask.png')))
            image.save(os.path.join(os.path.join(opt.input_dir, "resized_images", f'{os.path.basename(sequence_folder)}', f'{image_name}_image.png')))
        else:
            image = image.convert('RGBA')
            image = image.crop(bbox)
            image = image.resize((380, 380), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:] + (1 - image[:, :, 3:4]) * 1.0
            paste_position = ((512 - 380) // 2, (512 - 380) // 2)
            new_rgb_array = np.ones((512, 512, 3))
            new_rgb_array[paste_position[1]:paste_position[1]+380, 
                         paste_position[0]:paste_position[0]+380, :3] = image
            image = Image.fromarray((new_rgb_array * 255).astype(np.uint8))
            image.save(os.path.join(os.path.join(opt.input_dir, "resized_images", f'{os.path.basename(sequence_folder)}', f'{image_name}_image.png')))
        frames.append(image) 
        image = image.resize((518, 518), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255
        
        # Handle both RGB and RGBA images
        if image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, 3:]
        elif image.shape[-1] == 3:
            image = image[:, :, :3]
            
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

    imageio.mimwrite(video_path, frames, fps=10, quality=8)
    # Concatenate features from all frames
    pack = {
        'patchtokens': np.stack(all_patchtokens, axis=0).astype(np.float16),
        'clstoken': np.stack(all_clstokens, axis=0).astype(np.float16),
        'normalized': opt.normalize
    }
    features = np.concatenate([pack['patchtokens'], pack['clstoken'][:, None, :]], axis=1)

    os.makedirs(os.path.join(opt.output_dir, 'dinov2_cond_features'), exist_ok=True)
    np.savez_compressed(output_path, features=features)

def process_and_save_image(rgb_image_path, mask_path, output_path, bbox=None):
    """
    Read an RGB image and its mask, crop based on the mask, resize, and save as RGBA.
    
    Args:
        rgb_image_path: Path to the input RGB image
        mask_path: Path to the corresponding mask image
        output_path: Path to save the processed RGBA image
    """
    # Read the RGB image and mask
    image = Image.open(rgb_image_path)
    mask = Image.open(mask_path)
    
    # Convert mask to numpy array if not already
    mask_np = np.array(mask)
    
    # Handle different mask formats
    if len(mask_np.shape) == 3 and mask_np.shape[2] > 1:
        # If mask has multiple channels, use first channel
        mask_np = mask_np[..., 0]
    
    # Perform the crop and resize based on mask
    alpha = mask_np
    if bbox is None:
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    
    output = image.crop(bbox)
    mask = mask.crop(bbox)
    
    # First resize to 380x380
    output = output.resize((380, 380), Image.Resampling.LANCZOS)
    mask = mask.resize((380, 380), Image.Resampling.LANCZOS)

    rgb_array = np.array(output).astype(np.float32) / 255
    mask_array = np.array(mask).astype(np.float32) / 255
    
    # Calculate position to paste (center the 380x380 image)
    paste_position = ((512 - 380) // 2, (512 - 380) // 2)
    new_rgb_array = np.ones((512, 512, 3))
    new_mask_array = np.zeros((512, 512))
    
    # Paste the smaller image into the center of the larger canvas
    new_rgb_array[paste_position[1]:paste_position[1]+380, 
                 paste_position[0]:paste_position[0]+380, :3] = rgb_array
    new_mask_array[paste_position[1]:paste_position[1]+380, 
                  paste_position[0]:paste_position[0]+380] = mask_array
    
    new_rgb_array[:, :, :3] = new_rgb_array[:, :, :3] * new_mask_array[:, :, None] + (1 - new_mask_array[:, :, None])
    final_img = np.concatenate([new_rgb_array, new_mask_array[:, :, None]], axis=2)
    
    output = Image.fromarray((final_img * 255).astype(np.uint8), 'RGBA')
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.save(output_path)
    return output, bbox

def main(rank, world_size, opt):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    base_dir = os.path.dirname(os.path.normpath(opt.input_dir))
    os.makedirs(base_dir, exist_ok=True)
    opt.output_dir = base_dir

    # Load start frame indices if provided
    start_indices = {}
    sample_rates = {}
    if opt.start_frame_file:
        with open(opt.start_frame_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    sequence_name, start_idx, sample_rate = parts
                    start_indices[sequence_name] = int(start_idx)
                    sample_rates[sequence_name] = int(sample_rate)

    # Only rank 0 downloads the model first
    if rank == 0:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dist.barrier()
    if rank != 0:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    
    dinov2_model.eval().cuda(rank)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get all sequence folders
    sequence_folders = sorted([os.path.join(opt.input_dir, d) for d in os.listdir(opt.input_dir) 
                             if os.path.isdir(os.path.join(opt.input_dir, d))])
    
    # Distribute folders across GPUs
    start = len(sequence_folders) * rank // world_size
    end = len(sequence_folders) * (rank + 1) // world_size
    sequence_folders = sequence_folders[start:end]

    # Extract features
    for folder in tqdm(sequence_folders):
        try:
            sequence_name = os.path.basename(folder)
            start_idx = start_indices.get(sequence_name, 0)  # Default to 0 if not specified
            sample_rate = sample_rates.get(sequence_name, 1)
            process_sample(opt, folder, dinov2_model, transform, rank, start_idx, sample_rate)
        except Exception as e:
            print(f"Error processing {folder}: {e}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing sequence folders with video frames')
    parser.add_argument('--start_frame_file', type=str, default=None,
                        help='Text file containing sequence_name start_frame_idx pairs')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize features')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    main(0, 1, opt)

