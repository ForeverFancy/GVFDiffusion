import os
import cv2
import math
import random
import numpy as np
import torch
import json
import accelerate
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pytorch3d.transforms
import torch.nn.functional as F

import utils3d
from sparse.basic import SparseTensor

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    train=True,
    start_idx=-1,
    end_idx=-1,
    txt_file='',
    load_camera=0,
    cam_root_path=None,
    num_pts=4096,
    sample_timesteps=4,
    **kwargs,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    with open(txt_file) as f:
        all_files = f.read().splitlines()
    all_files = [os.path.join(data_dir, x) for x in all_files]

    if start_idx >= 0 and end_idx >= 0 and start_idx < end_idx:
        all_files = all_files[start_idx:end_idx]
    print("Loading files: ", len(all_files))

    dataset = VolumeDataset(
        image_size,
        all_files,
        shard=accelerator.process_index if train else 0,
        num_shards=accelerator.num_processes if train else 1,
        load_camera=load_camera,
        cam_root_path=cam_root_path,
        train=train,
        num_pts=num_pts,
        sample_timesteps=sample_timesteps,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, collate_fn=VolumeDataset.collate_fn
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, collate_fn=VolumeDataset.collate_fn
        )
    while True:
        yield from loader


class VolumeDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        load_camera=0,
        cam_root_path=None,
        train=True,
        num_timesteps=24,
        sample_timesteps=4,
        num_pts=4096,
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.resolution = resolution
        self.load_camera = load_camera
        self.cam_root_path = cam_root_path
        self.train = train
        self.num_timesteps = num_timesteps
        self.sample_timesteps = sample_timesteps
        self.num_pts = num_pts

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        canonical_timestep_idx = 0

        cam_root_path = self.cam_root_path
        data_dict = {"path": path, "canonical_timestep_idx": canonical_timestep_idx}
        root_path = os.path.join(cam_root_path, os.path.basename(path))
        feature_root_path = os.path.join(cam_root_path, os.path.basename(path))

        static_pc = torch.load(os.path.join(root_path, "objs/mesh_data/static_frame_vertices.pt"), map_location="cpu").to(torch.float32) # (8192, 3)
        moving_frame_deltas = torch.load(os.path.join(root_path, "objs/mesh_data/moving_frame_deltas.pt"), map_location="cpu").to(torch.float32) # (24, 8192, 3)

        moving_frame_pc = static_pc.unsqueeze(0).repeat(24, 1, 1) + moving_frame_deltas
        static_pc = moving_frame_pc[canonical_timestep_idx]
        moving_frame_deltas = moving_frame_pc - static_pc.unsqueeze(0)
        static_feat = load_feature(feature_root_path, vae_resolution=self.resolution, timestep_idx=canonical_timestep_idx)
        data_dict["static_feat"] = static_feat
        
        return (static_pc, moving_frame_deltas), data_dict

    @staticmethod
    def collate_fn(batch):
        # Unpack the batch
        pc_data = [b[0] for b in batch]  # List of (static_pc, moving_frame_deltas, target_gs_xyz_deltas)
        data_dicts = [b[1] for b in batch]  # List of data_dict

        # Handle point cloud data
        static_pcs = torch.stack([d[0] for d in pc_data])
        moving_deltas = torch.stack([d[1] for d in pc_data])

        # Special handling for static_feat (sparse tensor data)
        coords = []
        feats = []
        for i, data_dict in enumerate(data_dicts):
            static_feat = data_dict['static_feat']
            # Add batch dimension to coords
            coords.append(torch.cat([
                torch.full((static_feat['coords'].shape[0], 1), i, dtype=torch.int32),
                static_feat['coords']
            ], dim=-1))
            feats.append(static_feat['feats'])
        
        coords = torch.cat(coords, dim=0)
        feats = torch.cat(feats, dim=0)

        # Create output dictionary
        out_dict = {
            'static_feat': SparseTensor(
                coords=coords,
                feats=feats,
            )
        }

        # Handle other fields from data_dict
        for key in data_dicts[0].keys():
            if key != 'static_feat':
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

        return (static_pcs, moving_deltas), out_dict


def load_cam_pose(path):
    canonical_c2w = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, -1.0, -2.0], 
        [0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0]
    ]).to(torch.float32)
    canonical_c2w[:3, 1:3] *= -1

    cond_render_c2ws = []
    for i in range(5):
        camera_path = os.path.join(path, f'{i}', 'meta_data', 'timestep_00_view_00.json') 
        with open(camera_path) as json_file:
            frame = json.load(json_file)
            c2w = np.array(frame["matrix_world"])
            c2w[:3, 1:3] *= -1
            cond_render_c2ws.append(torch.tensor(c2w, dtype=torch.float32))
    
    cond_render_c2ws = torch.stack(cond_render_c2ws)
    return canonical_c2w, cond_render_c2ws

def load_img(path, image_size):
    original_image = Image.open(path).resize((image_size, image_size), Image.LANCZOS)
    im_data = np.array(original_image.convert("RGBA")).astype(np.float32)
    bg = np.array([1,1,1]).astype(np.float32)
    norm_data = im_data / 255.0
    alpha = norm_data[:, :, 3:4]
    blurred_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., np.newaxis]
    arr = norm_data[:,:,:3] * blurred_alpha + bg * (1 - blurred_alpha)
    image = torch.from_numpy(arr).permute(2, 0, 1)
    return image


def load_feature(path, vae_resolution=32, timestep_idx=None):
    DATA_RESOLUTION = 64
    feats_path = os.path.join(path, 'features', f'dinov2_vitl14_reg.npz') if timestep_idx is None else os.path.join(path, 'features_all_timesteps', f'{timestep_idx}', f'dinov2_vitl14_reg.npz')
    feats = np.load(feats_path, allow_pickle=True)
    coords = torch.tensor(feats['indices']).int()
    feats = torch.tensor(feats['patchtokens']).float()
    
    if vae_resolution != DATA_RESOLUTION:
        factor = DATA_RESOLUTION // vae_resolution
        coords = coords // factor
        coords, idx = coords.unique(return_inverse=True, dim=0)
        feats = torch.scatter_reduce(
            torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
            dim=0,
            index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
            src=feats,
            reduce='mean'
        )
    
    feat = {
        'coords': coords,
        'feats': feats,
    }
    return feat


def load_cam(root_path, timestep_idx, frame_idx, trans=np.array([0.0, 0.0, 0.0]), 
             scale=1.0, white_background=True, relative_transform=None, known_c2w=None):

    camera_path = os.path.join(root_path, 'meta_data', 'timestep_{:02d}_view_{:02d}.json'.format(timestep_idx, frame_idx)) 
    with open(camera_path) as json_file:
        frame = json.load(json_file)
        fovx = frame["x_fov"]

        image_path = os.path.join(root_path, 'imgs', 'timestep_{:02d}_view_{:02d}.png'.format(timestep_idx, frame_idx))        
        original_image = Image.open(image_path)
        im_data = np.array(original_image.convert("RGBA")).astype(np.float32)
        bg = np.array([1,1,1]).astype(np.float32) if white_background else np.array([0, 0, 0]).astype(np.float32)
        norm_data = im_data / 255.0
        alpha = norm_data[:, :, 3:4]
        blurred_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., np.newaxis]
        arr = norm_data[:,:,:3] * blurred_alpha + bg * (1 - blurred_alpha)
        image = torch.from_numpy(arr).permute(2, 0, 1)
        alpha = torch.from_numpy(alpha).permute(2, 0, 1)
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["matrix_world"]) if known_c2w is None else known_c2w
        if relative_transform is not None:
            c2w = relative_transform @ c2w
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fovy = focal2fov(fov2focal(fovx, original_image.size[0]), original_image.size[1])
    
    R = R
    T = T
    FoVx = fovx
    FoVy = fovy
    image_path = image_path

    image_width = original_image.size[1]
    image_height = original_image.size[0]

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

    return {"FoVx": fovx, "FoVy": fovy, "image_width": image_width, "image_height": image_height, "world_view_transform": world_view_transform, "projection_matrix": projection_matrix, "full_proj_transform": full_proj_transform, "camera_center": camera_center, "image": image, "c2w": c2w, "timestep_idx": timestep_idx, "frame_idx": frame_idx, "alpha": alpha, "intrinsics": intrinsics, "extrinsics": torch.tensor(w2c, dtype=torch.float32)}


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
