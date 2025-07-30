import os
import cv2
import math
import random
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import utils3d

import pytorch3d.transforms
from sparse.basic import SparseTensor

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    train=True,
    uncond_p=0,
    static_mean_file=None,
    static_std_file=None,
    deformation_mean_file=None,
    deformation_std_file=None,
    start_idx=-1,
    end_idx=-1,
    txt_file='',
    sample_timesteps=1,
    num_timesteps=24,
    **kwargs,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    with open(txt_file) as f:
        all_files = f.read().splitlines()
    
    if start_idx >= 0 and end_idx >= 0 and start_idx < end_idx:
        all_files = all_files[start_idx:end_idx]
    print("Loading files: ", len(all_files))

    dataset = VolumeDataset(
        data_dir,
        image_size,
        all_files,
        static_mean_file=static_mean_file,
        static_std_file=static_std_file,
        deformation_mean_file=deformation_mean_file,
        deformation_std_file=deformation_std_file,
        train=train,
        num_timesteps=num_timesteps,
        sample_timesteps=sample_timesteps,
        uncond_p=uncond_p,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, collate_fn=dataset.collate_fn if not train else None
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, collate_fn=dataset.collate_fn if not train else None
        )
    while True:
        yield from loader


class VolumeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        resolution,
        image_paths,
        train=True,
        uncond_p=0,
        static_mean_file=None,
        static_std_file=None,
        deformation_mean_file=None,
        deformation_std_file=None,
        num_timesteps=24,
        sample_timesteps=1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.local_images = image_paths
        self.train = train
        self.num_timesteps = num_timesteps
        self.sample_timesteps = sample_timesteps
        self.uncond_p = uncond_p
        self.static_mean = torch.load(static_mean_file).to(torch.float32) if static_mean_file is not None else 0
        self.static_std = torch.load(static_std_file).to(torch.float32) if static_std_file is not None else 1
        self.deformation_mean = torch.load(deformation_mean_file).to(torch.float32) if deformation_mean_file is not None else 0
        self.deformation_std = torch.load(deformation_std_file).to(torch.float32) if deformation_std_file is not None else 1

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx): 
        path = self.local_images[idx]
        
        # Load latent vectors
        try:
            latent_root = os.path.join(self.data_dir, "latents", path)
            latent_path = os.path.join(latent_root, "deformation_latent.pt")
            deformation_latent = torch.load(latent_path, map_location='cpu')
            
            mean = deformation_latent['latent_mean'].to(torch.float32) # Each is (T, 1024, 16)
            std = deformation_latent['latent_std'].to(torch.float32) # Each is (T, 1024, 16)
            
            latent_samples = mean + std * torch.randn_like(mean)
            latent_samples = (latent_samples - self.deformation_mean) / self.deformation_std
            
            static_latent_samples = deformation_latent['fps_sampled_gs_4096'].to(torch.float32) # Each is (4096, 14)
            static_latent = (static_latent_samples - self.static_mean) / self.static_std

            deformation_position_xyz = deformation_latent['fps_sampled_gs_1024'][..., :3].to(torch.float32) # Each is (1024, 3)
            deformation_position_xyz = (deformation_position_xyz - self.static_mean[..., :3]) / self.static_std[..., :3]
            
            # image_feature_root = os.path.join(self.data_dir, "dinov2_cond_features", path)
            image_feature_root = os.path.join("/mnt/blob/output/output_0108/4dvae9k_512latent16_64sparse_gs8_ema200k_latent", "dinov2_cond_features")


            img_feature_path = os.path.join(image_feature_root, f"{path}.npz")
            img_features = torch.from_numpy(np.load(img_feature_path, allow_pickle=True)['features']).to(torch.float32) # (T, 1+L, 1024), 1 is for cls token

            # Sample timesteps
            time_ind = np.random.default_rng().choice(self.num_timesteps, self.sample_timesteps, replace=False) if (self.train and self.sample_timesteps < self.num_timesteps) else np.arange(self.num_timesteps)
            latent_samples = latent_samples[time_ind]
            img_features = img_features[time_ind]

        except Exception as e:
            print(f"Error loading data for {path}: {e}")
            return self.__getitem__(random.randint(0, len(self.local_images) - 1))
        
        # Conditionally drop features during training
        if self.train:
            rand_p = random.random() 
            if rand_p < self.uncond_p:
                img_features = torch.zeros_like(img_features)
        
        data_dict = {"path": path, "cond_images": img_features, "static_latent": static_latent, "deformation_position_xyz": deformation_position_xyz}
        
        return latent_samples, data_dict

    @staticmethod
    def collate_fn(batch):
        # Unpack the batch
        latent_samples = [b[0] for b in batch]  # List of latent samples
        data_dicts = [b[1] for b in batch]  # List of data_dict

        # Stack latent samples
        latent_samples = torch.stack(latent_samples)

        # Special handling for static_latent (sparse tensor data)
        coords = []
        feats = []
        T = latent_samples.shape[1]  # Get number of timesteps
        for i, data_dict in enumerate(data_dicts):
            static_gs = data_dict['static_gs']
            # Repeat coords T times with batch index i*T to (i+1)*T-1
            base_coords = static_gs['coords']
            for t in range(T):
                coords.append(torch.cat([
                    torch.full((base_coords.shape[0], 1), i*T + t, dtype=torch.int32),
                    base_coords
                ], dim=-1))
            # Repeat features T times
            feats.append(static_gs['feats'].repeat(T, 1))
        
        coords = torch.cat(coords, dim=0)
        feats = torch.cat(feats, dim=0)

        # Create output dictionary
        out_dict = {
            'static_gs': SparseTensor(
                coords=coords,
                feats=feats,
            )
        }

        # Handle other fields from data_dict
        for key in data_dicts[0].keys():
            if key != 'static_gs':
                if isinstance(data_dicts[0][key], list):
                    # Handle list-type data (like cams)
                    out_dict[key] = [item if not torch.is_tensor(item) else item for d in data_dicts for item in d[key]]
                else:
                    # Handle regular tensor data
                    if not isinstance(data_dicts[0][key], str):
                        if isinstance(data_dicts[0][key], dict):
                            # Handle dictionary data
                            out_dict[key] = {
                                k: torch.stack([d[key][k] for d in data_dicts])
                                for k in data_dicts[0][key].keys()
                            }
                        else:
                            values = [torch.tensor(d[key]) if not torch.is_tensor(d[key]) else d[key] for d in data_dicts]
                            out_dict[key] = torch.stack(values)
                    else:
                        out_dict[key] = [d[key] for d in data_dicts]

        return latent_samples, out_dict


def sample_latent(latent):
    mean, logvar = torch.chunk(latent, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    x = mean + std * torch.randn(mean.shape)
    return x
