import os
import imageio
from tqdm import tqdm
import torch
import random
import numpy as np
from PIL import Image
from torch_cluster import fps
import torch.nn.functional as F
from torchvision import transforms
from kiui.cam import orbit_camera
import pytorch3d
import clip
from utils import logger
from .script_util import build_rotation
import shutil
from huggingface_hub import hf_hub_download, snapshot_download


# Model repository mapping
MODEL_REPOS = {
    "GVFDiffusion_v1.0": {
        "repo_id": "BwZhang/GaussianVariationFieldDiffusion",
        "revision": "main",
        "model_path": "ema_diffusion_0.9999_500000.pt",
        "vae_path": "ema_deformation_0.9999_200000.pt",
        "static_vae_path": "ema_static_vae_0.9999_200000.pt",
        "static_mean_path": "static_mean.pt",
        "static_std_path": "static_std.pt",
        "deformation_mean_path": "deformation_mean.pt",
        "deformation_std_path": "deformation_std.pt",
        "assets_dir": "assets"
    }
}


def align_gaussian_to_canonical(static_gs_model, canonical_image, canonical_alpha, intrinsics, static_vae, id, device, in_the_wild=True):
    # align azimuth
    best_azi = 0
    best_diff = 1e8
    best_scale_factor = 1.0

    elevation = 0
    radius = 2.0
    convert_mat = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    static_vae.renderers["MipGS"].pipe.use_mip_gaussian = False
    # Find best azimuth
    azimuths_range = np.arange(-180, 180, 1) if in_the_wild else np.arange(-180, 180, 90)
    for v, azi in tqdm(enumerate(azimuths_range), desc="Finding best azimuth"):
        cam_poses = orbit_camera(elevation, azi, radius=radius, opengl=True)
        cam_poses = convert_mat @ cam_poses

        cam_poses[:3, 1:3] *= -1 # invert up & forward direction
        extrinsics = np.linalg.inv(cam_poses)
        extrinsics = torch.from_numpy(extrinsics)

        res = static_vae.renderers["MipGS"].render(
                static_gs_model, 
                extrinsics.to(device), 
                intrinsics.to(device),
            )

        # Calculate object size from alpha mask
        alpha = res["alpha"].squeeze() # [H, W]
        rendered_mask = (alpha > 0.5).float()
        
        # Get bounding box of rendered object
        y_indices, x_indices = torch.where(rendered_mask > 0)
        if len(y_indices) == 0:
            continue
            
        rendered_height = y_indices.max() - y_indices.min()
        rendered_width = x_indices.max() - x_indices.min()
        rendered_size = max(rendered_height, rendered_width)

        # Calculate target size from canonical image alpha
        canonical_mask = (canonical_alpha > 0.5).float()
        y_indices, x_indices = torch.where(canonical_mask > 0)
        if len(y_indices) == 0:
            continue
            
        canonical_height = y_indices.max() - y_indices.min()
        canonical_width = x_indices.max() - x_indices.min()
        canonical_size = max(canonical_height, canonical_width)

        # Calculate scale factor
        scale_factor = canonical_size / rendered_size
        target_size = int(512 * scale_factor)
        
        # resize the rgb image according to the scale factor and then center crop it original size
        image = res["rgb"].clamp(0.0, 1.0)
        image = F.interpolate(image.unsqueeze(0), size=(target_size, target_size), mode='bicubic', align_corners=False).squeeze(0)
        # Center pad or crop to 512x512
        _, H, W = image.shape
        if H < 512 or W < 512:
            # Zero pad to 512x512
            pad_h = max(0, (512 - H) // 2)
            pad_w = max(0, (512 - W) // 2)
            image = F.pad(image, (pad_w, pad_w + (512 - W - 2*pad_w), 
                                pad_h, pad_h + (512 - H - 2*pad_h)), 
                         mode='constant', value=1.0)
        else:
            # Center crop to 512x512
            top = (H - 512) // 2
            left = (W - 512) // 2
            image = image[:, top:top+512, left:left+512]

        image = image.clamp(0.0, 1.0)
        l1_diff = ((image - canonical_image).abs()).mean()
        
        # Calculate CLIP similarity
        # Convert tensors to PIL images for CLIP preprocessing
        transform = transforms.ToPILImage()
        render_pil = transform(image.cpu())
        canonical_pil = transform(canonical_image.cpu())
        
        # Apply CLIP preprocessing
        render_clip = preprocess(render_pil).unsqueeze(0).to(device)
        canonical_clip = preprocess(canonical_pil).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            render_features = clip_model.encode_image(render_clip)
            canonical_features = clip_model.encode_image(canonical_clip)
            
            # Normalize features
            render_features = render_features / render_features.norm(dim=-1, keepdim=True)
            canonical_features = canonical_features / canonical_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            clip_similarity = (render_features @ canonical_features.T).item()
            clip_diff = 1.0 - clip_similarity
        
        # Combine all metrics
        diff = l1_diff + clip_diff * 0.2

        s_path = os.path.join(logger.get_dir(), 'align_gaussian_images', f'{id:03d}')
        os.makedirs(s_path, exist_ok=True)
        rgb_map = image.squeeze().permute(1, 2, 0).cpu()
        rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
        imageio.imwrite(
            os.path.join(s_path, f"render_{azi:03d}_diff_{l1_diff:.4f}_{clip_diff:.4f}.png"),
            rgb_map
        )

        if diff < best_diff:
            best_diff = diff
            best_azi = azi
            best_scale_factor = scale_factor
    print(f"\nID: {id} \tBest azimuth: {best_azi} \tBest scale factor: {best_scale_factor}")
    
    # Create rotation matrix for best azimuth
    # We want to rotate so that the best_azi view becomes the front view (azi=0)
    angle_rad = np.radians(-best_azi)
    rotation_matrix = torch.tensor([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    # Apply rotation to positions
    xyz = static_gs_model.get_xyz
    xyz = (rotation_matrix @ xyz.T).T
    static_gs_model.from_xyz(xyz)
    
    # Transform rotations
    rotations = static_gs_model.get_rotation
    rotations_mat = build_rotation(rotations).to(xyz.device)
    rotations_mat = rotation_matrix @ rotations_mat  # Note: removed the .T
    rotations = pytorch3d.transforms.matrix_to_quaternion(rotations_mat)
    static_gs_model.from_rotation(rotations)
    
    return static_gs_model, best_scale_factor


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


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def render_and_save_images(
    args,
    static_vae, 
    static_gs_model, 
    pred_delta, 
    model_kwargs, 
    valid_idx, 
    img_id, 
    accelerator, 
    scale_factors,
    save_dir='inference_images'
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
    os.makedirs(os.path.join(logger.get_dir(), save_dir), exist_ok=True)
    os.makedirs(os.path.join(logger.get_dir(), 'inference_videos'), exist_ok=True)

    B = pred_delta.shape[0]
    static_vae.renderers["MipGS"].pipe.use_mip_gaussian = True

    azimuths = np.arange(0, 360, 360 / 128)
    elevation = 0
    radius = 2.0
    
    cams = []
    convert_mat = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
    for timestep_idx in range(0, 32):
        for i, azi in enumerate(azimuths):
            cam_poses = orbit_camera(elevation, azi, radius=radius, opengl=True)
            cam_poses = convert_mat @ cam_poses
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            cam_poses[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = torch.from_numpy(np.linalg.inv(cam_poses))
            cams.append(w2c)
    
    for b in range(B):
        for t in range(32):
            for cam_idx in range(128):
                extrinsics = cams[cam_idx].to(accelerator.device)
                intrinsics = model_kwargs["cams"]["intrinsics"][0][0].to(accelerator.device)
                timestep_idx = t
                pred_delta_b = pred_delta[b][timestep_idx, :valid_idx[b]]
            
                res = static_vae.renderers["MipGS"].render(
                    static_gs_model[b], 
                    extrinsics, 
                    intrinsics, 
                    delta_pc=pred_delta_b
                )

                # Save rendered images
                s_path = os.path.join(logger.get_dir(), save_dir)
                os.makedirs(s_path, exist_ok=True)
                
                target_size = int(512 * scale_factors[b])
                image = res["rgb"].clamp(0.0, 1.0)
                
                output_image = image.clamp(0.0, 1.0)

                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu()
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                image = Image.fromarray(rgb_map).resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
                W, H = image.size
                if H < 512 or W < 512:
                    # Zero pad to 512x512
                    pad_h = max(0, (512 - H) // 2)
                    pad_w = max(0, (512 - W) // 2)
                    new_image = Image.new('RGB', (512, 512), (255, 255, 255))
                    new_image.paste(image, (pad_w, pad_h))
                    image = new_image
                else:
                    # Center crop to 512x512
                    left = (W - 512) // 2
                    top = (H - 512) // 2
                    image = image.crop((left, top, left + 512, top + 512))
                
                image.save(os.path.join(s_path, f"rank_{accelerator.process_index:02d}_render_{img_id+b:06d}_cam_{cam_idx:03d}_timesteps_{timestep_idx:02d}.png"))
        
        create_spiral_timeline_video(
            os.path.join(logger.get_dir(), save_dir),
            os.path.join(logger.get_dir(), 'inference_videos', f'video_{img_id+b}.mp4'),
            num_timesteps=32,
            views_per_timestep=128,
            fps=10,
            id=img_id+b
        )

def create_spiral_timeline_video(
    input_dir,
    output_path,
    num_timesteps=32,
    views_per_timestep=128,
    fps=10,
    id=0
):
    """
    Create a video that:
    1. First shows all timesteps from the frontal view
    2. Then shows spiral progression (simultaneous time and view rotation) multiple times
    
    Args:
        input_dir: Directory containing the rendered images
        output_path: Path for the output video
        num_timesteps: Number of timesteps in the sequence
        views_per_timestep: Number of views per timestep
        fps: Frames per second for the output video
        id: ID of the rendered image
    """
    
    frames = []
    
    # First loop: Show all timesteps from frontal view (v=0)
    for t in range(num_timesteps):
        image_path = os.path.join(
            input_dir,
            f"rank_00_render_{id:06d}_cam_000_timesteps_{t:02d}.png"
        )
        
        if os.path.exists(image_path):
            frame = imageio.imread(image_path)
            frames.append(frame)
        else:
            print(f"Warning: Image not found {image_path}")
    
    # Calculate views per 90-degree segment
    views_per_segment = views_per_timestep // 4 # 6 views per 90 degrees if total is 24
    
    # Loop through each 90-degree segment
    for segment in range(4):
        start_view = segment * views_per_segment
        end_view = (segment + 1) * views_per_segment

        # Calculate total frames needed for this segment
        total_frames = min(num_timesteps, views_per_segment)
        
        # Create spiral progression within this segment
        for frame_idx in range(total_frames):
            # Calculate timestep and view indices
            t = frame_idx % num_timesteps
            v = (start_view + (frame_idx % views_per_segment)) % views_per_timestep
            
            image_path = os.path.join(
                input_dir,
                f"rank_00_render_{id:06d}_cam_{v:03d}_timesteps_{t:02d}.png"
            )
            
            if os.path.exists(image_path):
                frame = imageio.imread(image_path)
                frames.append(frame)
            else:
                print(f"Warning: Image not found {image_path}")
    
    # Write video using imageio
    imageio.mimwrite(
        output_path, 
        frames, 
        fps=fps, 
        quality=8,
        codec='libx264',
        output_params=['-pix_fmt', 'yuv420p']
    )


def download_model_files(model_name):
    """Download model files from Hugging Face Hub."""
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REPOS.keys())}")
    
    model_info = MODEL_REPOS[model_name]
    downloaded_files = {}
    
    try:
        # Download model checkpoint
        downloaded_files["ckpt"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["model_path"],
            revision=model_info["revision"]
        )
        
        # Download VAE checkpoint
        downloaded_files["vae_ckpt"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["vae_path"],
            revision=model_info["revision"]
        )
        
        # Download Static VAE checkpoint
        downloaded_files["static_vae_ckpt"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["static_vae_path"],
            revision=model_info["revision"]
        )
        
        # Download mean and std files
        downloaded_files["static_mean"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["static_mean_path"],
            revision=model_info["revision"]
        )
        
        downloaded_files["static_std"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["static_std_path"],
            revision=model_info["revision"]
        )
        
        downloaded_files["deformation_mean"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["deformation_mean_path"],
            revision=model_info["revision"]
        )
        
        downloaded_files["deformation_std"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["deformation_std_path"],
            revision=model_info["revision"]
        )
        
    except Exception as e:
        print(f"Error downloading files for {model_name}: {e}")
        raise
    
    return downloaded_files


def download_example_assets(model_name, local_dir="./assets"):
    """Download example assets from Hugging Face Hub."""
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REPOS.keys())}")
    
    model_info = MODEL_REPOS[model_name]
    
    try:
        # Create local assets directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download assets directory from the repo
        print(f"Downloading example assets from {model_info['repo_id']}...")
        
        # Use snapshot_download to download the assets directory
        repo_dir = snapshot_download(
            repo_id=model_info["repo_id"],
            revision=model_info["revision"],
            allow_patterns=f"{model_info['assets_dir']}/**",
        )
        
        # Copy the assets to the local directory
        assets_source = os.path.join(repo_dir, model_info["assets_dir"])
        
        # Copy all contents from assets_source to local_dir
        for item in os.listdir(assets_source):
            source_path = os.path.join(assets_source, item)
            dest_path = os.path.join(local_dir, item)
            
            if os.path.isdir(source_path):
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)
                
        print(f"Example assets downloaded successfully to {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"Error downloading example assets for {model_name}: {e}")
        raise
