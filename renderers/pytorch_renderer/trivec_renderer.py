from abc import abstractmethod
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from torch_scatter import segment_coo
from .apparatus import *
from tqdm import tqdm
import utils3d.torch
from .sh import eval_sh_bases


class TrivecRenderingSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    used_rank : int
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    debug : bool


class TrivecRenderer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def sample_ray_cvrg_cuda(self, rays_o, rays_d, voxels, units, aabb, stepSize):
        radius = rays_o[0].norm().item()
        near, far = radius - 2, radius + 2
        rays_o = rays_o.view(-1, 3).contiguous()
        rays_d = rays_d.view(-1, 3).contiguous()
        ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_cvrg(
            rays_o, rays_d, voxels, units, aabb[0], aabb[1], near, far, stepSize
        )
               
        ray_pts = ray_pts[mask_valid]
        ray_id = ray_id[mask_valid]
        step_id = step_id[mask_valid]
        
        return ray_pts, t_min, ray_id, step_id

    def sample_2_tensoRF_cvrg_hier(self, xyz_sampled, aabb, positions, voxel_indices, units, trivec_dim):
        # to find the grid indeces of each sample point's TopK tensors for feature interpolation and aggregation 
        local_unit = units / trivec_dim
        local_range = 0.5 * units / trivec_dim * (trivec_dim - 1)
        local_dims = torch.tensor([trivec_dim-1, trivec_dim-1, trivec_dim-1], dtype=torch.int64, device=xyz_sampled.device)
        tensoRF_count = torch.ones((positions.shape[0],), dtype=torch.int8, device=positions.device)
        tensoRF_topindx = torch.arange(positions.shape[0], dtype=torch.int32, device=positions.device).unsqueeze(1)
        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_hier_cuda.sample_2_tensoRF_cvrg_hier(
            xyz_sampled.contiguous(), aabb[0], aabb[1], units, local_unit, local_range, local_dims,
            voxel_indices, tensoRF_count, tensoRF_topindx, positions, 1, True
        )
        return local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id

    def ind_intrp_line_map_batch_prod(self, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        return (density_lines[0][tensoRF_id, :, local_gindx_s[..., 0]] * local_gweight_s[:, None, 0] + density_lines[0][tensoRF_id, :, local_gindx_l[..., 0]] * local_gweight_l[:, None, 0]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., 1]] * local_gweight_s[:, None, 1] + density_lines[1][tensoRF_id, :, local_gindx_l[..., 1]] * local_gweight_l[:, None, 1]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., 2]] * local_gweight_s[:, None, 2] + density_lines[2][tensoRF_id, :, local_gindx_l[..., 2]] * local_gweight_l[:, None, 2])
    
    def compute_feature(self, trivecs, densities, colors, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        feature = (trivecs[tensoRF_id, :, 0, local_gindx_s[..., 0]] * local_gweight_s[:, None, 0] + trivecs[tensoRF_id, :, 0, local_gindx_l[..., 0]] * local_gweight_l[:, None, 0]) * \
                  (trivecs[tensoRF_id, :, 1, local_gindx_s[..., 1]] * local_gweight_s[:, None, 1] + trivecs[tensoRF_id, :, 1, local_gindx_l[..., 1]] * local_gweight_l[:, None, 1]) * \
                  (trivecs[tensoRF_id, :, 2, local_gindx_s[..., 2]] * local_gweight_s[:, None, 2] + trivecs[tensoRF_id, :, 2, local_gindx_l[..., 2]] * local_gweight_l[:, None, 2])
        density = (feature * densities[tensoRF_id]).sum(dim=1)
        color = (feature.unsqueeze(-1) * colors[tensoRF_id]).sum(dim=1)
        return density, color
                  
    def forward(self, positions, trivecs, densities, shs = None, colors_precomp = None, depths = None, aabb = None, patch_mask = None):
        # Get the voxel grid
        n_pts = positions.shape[0]
        n_rays = self.raster_settings.image_height * self.raster_settings.image_width
        depth = depths[0].item()
        assert torch.all(depths == depth), "All depths must be the same"
        voxel_res = 2**depth
        aabb = torch.stack([aabb[0:3], aabb[0:3] + aabb[3:6]], dim=0)
        units = (aabb[1] - aabb[0]) / voxel_res
        step_size = 0.5 * units.min().item() / trivecs.shape[-1]
        valid_indices = (positions * voxel_res).long()
        valid_indices = valid_indices[:, 0] * voxel_res * voxel_res + valid_indices[:, 1] * voxel_res + valid_indices[:, 2]
        voxels = torch.zeros(voxel_res * voxel_res * voxel_res, dtype=torch.bool, device=positions.device)
        voxels[valid_indices] = True
        voxels = voxels.view(voxel_res, voxel_res, voxel_res)
        voxel_indices = torch.zeros(voxel_res * voxel_res * voxel_res, dtype=torch.int32, device=positions.device)
        voxel_indices[valid_indices] = torch.arange(valid_indices.shape[0], dtype=torch.int32, device=positions.device)
        voxel_indices = voxel_indices.view(voxel_res, voxel_res, voxel_res)
        positions = positions * (aabb[1] - aabb[0]).view(1, 3) + aabb[0].view(1, 3)

        # Sample rays
        rays_o = self.raster_settings.campos.view(1, 3).expand(n_rays, 3)
        src_coord = (2 * utils3d.torch.image_uv(self.raster_settings.image_height, self.raster_settings.image_width) - 1).reshape(-1,2).float().to(rays_o)
        rays_d = F.normalize(torch.stack([
            src_coord[:, 0] * self.raster_settings.tanfovx,
            src_coord[:, 1] * self.raster_settings.tanfovy,
            torch.ones_like(src_coord[:, 0])
        ], dim=-1))
        rays_d = (self.raster_settings.viewmatrix[None, :3, :3] @ rays_d[..., None]).squeeze(-1)
        if patch_mask is not None:
            patch_mask = patch_mask.view(-1)
            rays_o = rays_o[patch_mask]
            rays_d = rays_d[patch_mask]
            n_rays = rays_o.shape[0]

        if colors_precomp is None:
            # self.raster_settings.sh_degree = 0
            active_sh_channels = (self.raster_settings.sh_degree + 1) ** 2
            dirs = F.normalize(positions - self.raster_settings.campos.view(1, 3))
            colors_precomp = \
                (eval_sh_bases(self.raster_settings.sh_degree, dirs).reshape(n_pts, 1, -1, 1) * shs[:, :, :active_sh_channels]).sum(dim=-2)

        # patched rendering
        rgb_patch, depth_patch, alpha_patch = [], [], []
        patch_size = 100000
        for i in range(0, n_rays, patch_size):
            inner_size = min(patch_size, n_rays - i)
            ray_pts, t_min, ray_id, step_id = self.sample_ray_cvrg_cuda(rays_o[i:i+inner_size], rays_d[i:i+inner_size], voxels, units, aabb, step_size)
            if ray_id is None or len(ray_id) == 0:
                rgb_patch.append(torch.zeros([inner_size, 3], device=positions.device, dtype=torch.float32) + self.raster_settings.bg.unsqueeze(0))
                depth_patch.append(torch.zeros([inner_size], device=positions.device, dtype=torch.float32))
                alpha_patch.append(torch.zeros([inner_size], device=positions.device, dtype=torch.float32))
                continue

            # Sample trivecs
            local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg_hier(ray_pts, aabb, positions, voxel_indices, units, trivecs.shape[-1])
            sigma, color = self.compute_feature(trivecs, densities, colors_precomp, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id)
            density = F.softplus(sigma.flatten() - self.raster_settings.density_shift * 10) * min(1 / (1 - self.raster_settings.density_shift), 25)
            alpha = 1 - torch.exp(-density * step_size)
            weights, bg_weight = Alphas2Weights.apply(alpha, ray_id, inner_size)
            rgb = F.sigmoid(color)

            rgb_map = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([inner_size, 3], device=weights.device, dtype=torch.float32),
                reduce='sum')
            rgb_map += bg_weight.unsqueeze(-1) * self.raster_settings.bg.unsqueeze(0)

            z_val = t_min[ray_id] + step_id * step_size
            depth_map = segment_coo(
                src=(weights.unsqueeze(-1) * z_val.unsqueeze(-1)),
                index=ray_id,
                out=torch.zeros([inner_size, 1], device=weights.device, dtype=torch.float32),
                reduce='sum').squeeze(-1)
            alpha_map = 1 - bg_weight

            rgb_patch.append(rgb_map)
            depth_patch.append(depth_map)
            alpha_patch.append(alpha_map)

            #  DEBUG
            # dbg_ray_id = 241932
            # if dbg_ray_id in list(range(i, i+inner_size)):
            #     dbg_ray_id = dbg_ray_id - i
            #     dbg_position = torch.zeros([16384, 3], device=positions.device, dtype=torch.float32)
            #     dbg_density = torch.zeros([16384], device=positions.device, dtype=torch.float32)
            #     dbg_color = torch.zeros([16384, 3], device=positions.device, dtype=torch.float32)
            #     dbg_weight = torch.zeros([16384], device=positions.device, dtype=torch.float32)
            #     step_id += (t_min[ray_id] / step_size).round().long()
            #     ray_mask = ray_id == dbg_ray_id
            #     dbg_position[step_id[ray_mask]] = ray_pts[ray_mask]
            #     dbg_density[step_id[ray_mask]] = F.softplus(sigma[ray_mask])
            #     dbg_color[step_id[ray_mask]] = rgb[ray_mask]
            #     dbg_weight[step_id[ray_mask]] = weights[ray_mask]
            #     torch.save(dbg_position, 'dbg_pytorch_position.pt')
            #     torch.save(dbg_density, 'dbg_pytorch_density.pt')
            #     torch.save(dbg_color, 'dbg_pytorch_color.pt')
            #     torch.save(dbg_weight, 'dbg_pytorch_weight.pt')

        rgb_map = torch.cat(rgb_patch, dim=0)
        depth_map = torch.cat(depth_patch, dim=0)
        alpha_map = torch.cat(alpha_patch, dim=0)

        if patch_mask is not None:
            rgb_patch, depth_patch, alpha_patch = rgb_map, depth_map, alpha_map
            rgb_map = torch.zeros([patch_mask.shape[0], 3], device=positions.device, dtype=torch.float32)
            depth_map = torch.zeros([patch_mask.shape[0]], device=positions.device, dtype=torch.float32)
            alpha_map = torch.zeros([patch_mask.shape[0]], device=positions.device, dtype=torch.float32)
            rgb_map[patch_mask] = rgb_patch
            depth_map[patch_mask] = depth_patch
            alpha_map[patch_mask] = alpha_patch

        rgb_map = rgb_map.transpose(0, 1).reshape(3, self.raster_settings.image_height, self.raster_settings.image_width)
        depth_map = depth_map.reshape(self.raster_settings.image_height, self.raster_settings.image_width)
        alpha_map = alpha_map.reshape(self.raster_settings.image_height, self.raster_settings.image_width)

        return rgb_map, depth_map, alpha_map
        
