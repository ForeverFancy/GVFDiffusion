#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
from representations.gaussian import GaussianModel
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict


# def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = R.transpose()
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = np.linalg.inv(Rt)
#     cam_center = C2W[:3, 3]
#     cam_center = (cam_center + translate) * scale
#     C2W[:3, 3] = cam_center
#     Rt = np.linalg.inv(C2W)
#     return np.float32(Rt)

# def getProjectionMatrix(znear, zfar, fovX, fovY):
#     tanHalfFovY = math.tan((fovY / 2))
#     tanHalfFovX = math.tan((fovX / 2))

#     top = tanHalfFovY * znear
#     bottom = -top
#     right = tanHalfFovX * znear
#     left = -right

#     P = torch.zeros(4, 4)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)
#     return P

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, delta_pc = None, detach_static=False, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # lazy import
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if pipe.use_mip_gaussian:
        kernel_size = pipe.kernel_size
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
    else:
        print(globals())
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    shs = None
    if delta_pc is not None:
        means3D = pc.get_xyz_with_delta(delta_pc[..., :3], detach=detach_static)
        scales = pc.get_scaling_with_delta(delta_pc[..., 3:6], detach=detach_static)
        rotations = pc.get_rotation_with_delta(delta_pc[..., 6:10], detach=detach_static)
        if delta_pc.shape[1] > 10:
            shs = pc.get_features_with_delta(delta_pc[..., 10:13].unsqueeze(1), detach=detach_static)
            opacity = pc.get_opacity_with_delta(delta_pc[..., 13:], detach=detach_static)
    else:
        means3D = pc.get_xyz
        scales = pc.get_scaling
        rotations = pc.get_rotation
        opacity = pc.get_opacity
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features if shs is None else shs
    else:
        colors_precomp = override_color

    if delta_pc is not None:
        means3D = means3D.to(delta_pc.dtype)
        means2D = means2D.to(delta_pc.dtype)
        shs = shs.to(delta_pc.dtype)
        opacity = opacity.to(delta_pc.dtype)
        scales = scales.to(delta_pc.dtype)
        rotations = rotations.to(delta_pc.dtype)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return edict({"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})



class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "use_mip_gaussian" : False,
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(self, gausssian, extrinsics, intrinsics, delta_pc = None, detach_static=False, colors_overwrite=None, patch_mask=None):
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            rgb (torch.Tensor): (3, H, W) rendered rgb
            depth (torch.Tensor): (H, W) rendered depth
            alpha (torch.Tensor): (H, W) rendered alpha
            aux (dict): auxiliary data
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")



        # NOTE: here utils3d use standard OpenGL convention, where the camera is looking towards -z axis.
        # However, the voxel representation uses special convention, where the camera is looking towards +z axis.
        # Therefore, we need to flip the z axes of the camera.
        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })
        
        # R = extrinsics[:3,:3].t()
        # T = extrinsics[:3, 3]
        
        # intrinsics[0, 0] = intrinsics[0, 0] * 2
        # intrinsics[0, 2] = intrinsics[0, 2] * 2  - 1
        # intrinsics[1, 1] = intrinsics[1, 1] * 2 
        # intrinsics[1, 2] = intrinsics[1, 2] * 2  - 1

        # fovx = 2 * math.atan(1 / intrinsics[0, 0])
        # fovy = 2 * math.atan(1 / intrinsics[1, 1])

        # world_view_transform = torch.tensor(getWorld2View2(R.detach().cpu().numpy(), T.detach().cpu().numpy())).transpose(0, 1)
        # projection_matrix = getProjectionMatrix(znear=near, zfar=far, fovX=fovx, fovY=fovy).transpose(0,1)
        # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        # camera_center = world_view_transform.inverse()[3, :3]
        
        # camera_dict2 = edict({
        #     "image_height": resolution * ssaa,
        #     "image_width": resolution * ssaa,
        #     "FoVx": fovx,
        #     "FoVy": fovy,
        #     "znear": near,
        #     "zfar": far,
        #     "world_view_transform": world_view_transform,
        #     "projection_matrix": projection_matrix,
        #     "full_proj_transform": full_proj_transform,
        #     "camera_center": camera_center
        # })

        # Render
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, delta_pc=delta_pc, detach_static=detach_static, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bicubic', align_corners=False, antialias=True).squeeze()
            if 'feature' in render_ret:
                render_ret.feature = F.interpolate(render_ret.feature[None], size=(resolution, resolution), mode='bicubic', align_corners=False, antialias=True).squeeze()
            # render_ret.depth = F.interpolate(render_ret.depth[None, None], size=(resolution, resolution), mode='bicubic', align_corners=False, antialias=True).squeeze()
            # render_ret.alpha = F.interpolate(render_ret.alpha[None, None], size=(resolution, resolution), mode='bicubic', align_corners=False, antialias=True).squeeze()

        ret = edict({
            'rgb': render_ret['render']
        })
        return ret
