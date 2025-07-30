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

from sparse.basic import SparseTensor

def load_data(
    *,
    data_dir,
    batch_size,
    deterministic=False,
    load_camera=0,
    num_timesteps=24,
    start_idx=-1,
    end_idx=-1,
    txt_file='',
    in_the_wild=True,
    **kwargs,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    with open(txt_file) as f:
        all_files = f.read().splitlines()
    
    if start_idx >= 0 and end_idx >= 0 and start_idx < end_idx:
        all_files = all_files[start_idx:end_idx]
    all_files = [(line.split(" ")[0], int(line.split(" ")[1])) for line in all_files] if in_the_wild else [(line, 0) for line in all_files]
    print("Loading files: ", len(all_files))

    dataset = VolumeDataset(
        data_dir,
        all_files,
        shard=0,
        num_shards=1,
        in_the_wild=in_the_wild,
        load_camera=load_camera,
        num_timesteps=num_timesteps,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True,
        )
    while True:
        yield from loader


class VolumeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_paths,
        shard=0,
        num_shards=1,
        load_camera=0,
        in_the_wild=True,
        num_timesteps=24,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.local_images = image_paths[shard:][::num_shards]
        self.in_the_wild = in_the_wild
        self.num_timesteps = num_timesteps
        self.load_camera = load_camera

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # Load image features
        name, canonical_timestep_idx = self.local_images[idx]
        image_feature_root = os.path.join(self.data_dir, "dinov2_cond_features")
        img_feature_path = os.path.join(image_feature_root, f"{name}.npz")
        img_features = torch.from_numpy(np.load(img_feature_path, allow_pickle=True)['features']).to(torch.float32) # (T, 1+L, 1024), 1 is for cls token

        # Sample timesteps
        time_ind = np.arange(self.num_timesteps)
        img_features = img_features[time_ind]
        
        data_dict = {"path": name, "cond_images": img_features}
        
        # Define the specific camera angles (in degrees)
        from kiui.cam import orbit_camera
        azimuths = range(0, 360, 360 // self.load_camera)
        elevation = 0
        radius = 2.0

        cams = []
        convert_mat = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
        for timestep_idx in range(0, self.num_timesteps):
            for i, azi in enumerate(azimuths):
                cam_poses = orbit_camera(elevation, azi, radius=radius, opengl=True)
                cam_poses = convert_mat @ cam_poses
                cams.append(load_cam(timestep_idx, i, known_c2w=cam_poses))
        
        loading_cams = {"extrinsics": torch.stack([cam["extrinsics"] for cam in cams]),
                        "intrinsics": torch.stack([cam["intrinsics"] for cam in cams]),
                        "timestep_idx": torch.stack([torch.tensor(cam["timestep_idx"]) for cam in cams]),
                        "frame_idx": torch.stack([torch.tensor(cam["frame_idx"]) for cam in cams])}
        data_dict["cams"] = loading_cams
        data_dict["canonical_path"] = os.path.join(self.data_dir, 'frames/resized_images', name, f'{canonical_timestep_idx:06d}_image.png')
        if self.in_the_wild:
            data_dict["mask_path"] = os.path.join(self.data_dir, 'frames/resized_images', name, f'{canonical_timestep_idx:06d}_mask.png')
        
        return data_dict

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


def load_cam(timestep_idx, frame_idx, image_size=512, trans=np.array([0.0, 0.0, 0.0]), 
             scale=1.0, relative_transform=None, known_c2w=None):

    fovx = np.radians(49.1).astype(np.float32)
    
    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = known_c2w
    if relative_transform is not None:
        c2w = relative_transform @ c2w
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    fovy = focal2fov(fov2focal(fovx, image_size), image_size)
    
    R = R
    T = T
    FoVx = fovx
    FoVy = fovy

    zfar = 100.0
    znear = 0.01

    trans = trans
    scale = scale

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    # build intrinsics
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fovx), torch.tensor(fovy))

    return {"FoVx": fovx, "FoVy": fovy, "image_width": image_size, "image_height": image_size, "world_view_transform": world_view_transform, "projection_matrix": projection_matrix, "full_proj_transform": full_proj_transform, "camera_center": camera_center, "c2w": c2w, "timestep_idx": timestep_idx, "frame_idx": frame_idx, "intrinsics": intrinsics, "extrinsics": torch.tensor(w2c, dtype=torch.float32)}


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
