from typing import *
import copy
import torch
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from representations.gaussian import GaussianModel
from renderers import GaussianRenderer

from sparse import SparseTensor, sparse_cat
from utils.loss_util import l1_loss, l2_loss, ssim, lpips
from .utils import hammersley_sequence, unwrap_dist


_DEFAULT_GAUSSIAN_LR_CONFIG = {
    "_xyz" : 1.0,
    "_features_dc": 0.0025,
    "_opacity": 0.05,
    "_scaling": 0.005,
    "_rotation": 0.001
}

_DEFAULT_RF_CP_CFG = {
    'rank': 8,
    'dim': 8,
}

_DEFAULT_GS_CFG = {
    'lr': _DEFAULT_GAUSSIAN_LR_CONFIG,
    'perturb_offset': False,
    'reg_mode': 'invoxel',
    'voxel_size': 1.1,
    'num_gaussians': 8,
    'scaling_bias': 0.01,
    'opacity_bias': 0.1,
    'scaling_activation': 'exp',
}

_DEFAULT_MIPGS_CFG = {
    'lr': _DEFAULT_GAUSSIAN_LR_CONFIG,
    'perturb_offset': False,
    'reg_mode': 'invoxel',
    'voxel_size': 1.1,
    'num_gaussians': 8,
    '2d_filter_kernel_size': 0.1,
    '3d_filter_kernel_size': 0.0,
    'scaling_bias': 0.01,
    'opacity_bias': 0.1,
    'scaling_activation': 'exp',
}

_DEFAULT_CONFIG = {
    'RF-CP': _DEFAULT_RF_CP_CFG,
    'GS': _DEFAULT_GS_CFG,
    'MipGS': _DEFAULT_MIPGS_CFG,
}


class SparseVAE:
    """
    A framework for training a VAE model on sparse data.

    Args:
        backbones: Dictionary of backbones.
    """
    def __init__(
        self,
        backbones,
        resolution=64,
        representation_config={},
        loss_type='l1',
        lambda_ssim=0.2,
        lambda_lpips=0.2,
        lamda_kl=1e-6,
        regularizations={},
        mem_ratio=1.0,
    ):
        assert 'vae' in backbones.keys(), 'A VAE backbone must be provided.'
        self.backbones = backbones
        self.resolution = resolution
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lamda_kl = lamda_kl
        self.rep_config = {}
        self.mem_ratio = mem_ratio
        for k, v in representation_config.items():
            self.rep_config[k] = _DEFAULT_CONFIG[k].copy()
            self.rep_config[k].update(v)
        self.regularizations = regularizations

        self._init_renderer()
        self._calc_layout(self.rep_config)
        
        if 'GS' in self.rep_config and self.rep_config['GS']['perturb_offset']:
            unwrap_dist(self.backbones['vae']).register_buffer('GS_perturbation', self._build_perturbation(self.rep_config['GS']['num_gaussians'], self.rep_config['GS']['reg_mode']))
        if 'MipGS' in self.rep_config and self.rep_config['MipGS']['perturb_offset']:
            unwrap_dist(self.backbones['vae']).register_buffer('MipGS_perturbation', self._build_perturbation(self.rep_config['MipGS']['num_gaussians'], self.rep_config['MipGS']['reg_mode']))
        
    def get_phases(self, step):
        return ['vae']
    
    def _build_perturbation(self, num_gaussians, reg_mode):
        offsets = [hammersley_sequence(3, i, num_gaussians) for i in range(num_gaussians)]
        offsets = torch.tensor(offsets).float() - 0.5
        if reg_mode == 'invoxel':
            pass
        elif reg_mode == 'soft_invoxel':
            offsets = offsets / 0.5 / self.rep_config['MipGS']['voxel_size']
        perturbation = torch.atanh(offsets)
        return perturbation.to(self.backbones['vae'].device)
    
    def to_representation(self, x):
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            dictionary of lists of representations.
        """
        ret = {k: [] for k in self.rep_config.keys()}
        for i in range(x.shape[0]):
                
            if 'GS' in self.rep_config:
                representation = GaussianModel(
                    sh_degree=0,
                    aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                    scaling_bias = self.rep_config['GS']['scaling_bias'],
                    opacity_bias = self.rep_config['GS']['opacity_bias'],
                    scaling_activation = self.rep_config['GS']['scaling_activation']
                )
                xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
                for k, v in self.layouts['GS'].items():
                    if k == '_xyz':
                        offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                        offset = offset * self.rep_config['GS']['lr'][k]
                        if self.rep_config['GS']['perturb_offset']:
                            offset = offset + unwrap_dist(self.backbones['vae']).GS_perturbation
                        if self.rep_config['GS']['reg_mode'] == 'invoxel':
                            offset = torch.tanh(offset) / self.resolution
                        elif self.rep_config['GS']['reg_mode'] == 'soft_invoxel':
                            offset = torch.tanh(offset) / self.resolution * 0.5 * 1.25
                        _xyz = xyz.unsqueeze(1) + offset
                        setattr(representation, k, _xyz.flatten(0, 1))
                    else:
                        feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                        feats = feats * self.rep_config['GS']['lr'][k]
                        setattr(representation, k, feats)
                ret['GS'].append(representation)
                
            if 'MipGS' in self.rep_config:
                representation = GaussianModel(
                    sh_degree=0,
                    aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                    mininum_kernel_size = self.rep_config['MipGS']['3d_filter_kernel_size'],
                    scaling_bias = self.rep_config['MipGS']['scaling_bias'],
                    opacity_bias = self.rep_config['MipGS']['opacity_bias'],
                    scaling_activation = self.rep_config['MipGS']['scaling_activation']
                )
                xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
                for k, v in self.layouts['MipGS'].items():
                    if k == '_xyz':
                        offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                        offset = offset * self.rep_config['MipGS']['lr'][k]
                        if self.rep_config['MipGS']['perturb_offset']:
                            offset = offset + unwrap_dist(self.backbones['vae']).MipGS_perturbation
                        if self.rep_config['MipGS']['reg_mode'] == 'invoxel':
                            offset = torch.tanh(offset) / self.resolution
                        elif self.rep_config['MipGS']['reg_mode'] == 'soft_invoxel':
                            offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['MipGS']['voxel_size']
                        _xyz = xyz.unsqueeze(1) + offset
                        setattr(representation, k, _xyz.flatten(0, 1))
                    else:
                        feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                        feats = feats * self.rep_config['MipGS']['lr'][k]
                        setattr(representation, k, feats)
                ret['MipGS'].append(representation)

        return ret
    
    def get_renderer(self, type, rendering_options):
        if type == 'GS':
            renderer = GaussianRenderer(rendering_options)
            return renderer
        elif type == 'MipGS':
            renderer = GaussianRenderer(rendering_options)
            renderer.pipe.use_mip_gaussian = True
            renderer.pipe.kernel_size = self.rep_config['MipGS']['2d_filter_kernel_size']
            return renderer
        else:
            raise ValueError(f"Invalid representation type: {type}")
    
    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : (1.0, 1.0, 1.0)}
        self.renderers = edict({k: self.get_renderer(k, rendering_options) for k in self.rep_config.keys()})
    
    def _calc_layout(self, rep_config):
        self.layouts ={}
        for k, v in rep_config.items():
            if k == 'RF-CP':
                self.layouts['RF-CP'] = {
                    'trivec': {'shape': (v['rank'], 3, v['dim']), 'size': v['rank'] * 3 * v['dim']},
                    'density': {'shape': (v['rank'],), 'size': v['rank']},
                    'features_dc': {'shape': (v['rank'], 1, 3), 'size': v['rank'] * 3},
                }
            elif k in ['GS', 'MipGS']:
                self.layouts[k] = {
                    '_xyz' : {'shape': (v['num_gaussians'], 3), 'size': v['num_gaussians'] * 3},
                    '_features_dc' : {'shape': (v['num_gaussians'], 1, 3), 'size': v['num_gaussians'] * 3},
                    '_scaling' : {'shape': (v['num_gaussians'], 3), 'size': v['num_gaussians'] * 3},
                    '_rotation' : {'shape': (v['num_gaussians'], 4), 'size': v['num_gaussians'] * 4},
                    '_opacity' : {'shape': (v['num_gaussians'], 1), 'size': v['num_gaussians']},
                }
            else:
                raise ValueError(f"Invalid representation type: {k}")
            
        self.layouts = edict(self.layouts)
        start = 0
        for k in list(self.layouts.keys()):
            for kk, vv in self.layouts[k].items():
                vv['range'] = (start, start + vv['size'])
                start += vv['size']
                
    def get_regularization_loss(self, x, reps):
        loss = 0.0
        terms = {}
        for k, v in reps.items():
            if not k in self.regularizations:
                continue
            if k == 'RF-CP':
                pass
            elif k in ['GS', 'MipGS']:
                if 'lambda_vol' in self.regularizations[k]:
                    scales = torch.cat([g.get_scaling for g in v], dim=0)   # [N x 3]
                    volume = torch.prod(scales, dim=1)  # [N]
                    terms[f'reg_{k}_vol'] = volume.mean()
                    loss = loss + self.regularizations[k]['lambda_vol'] * terms[f'reg_{k}_vol']
                if 'lambda_opacity' in self.regularizations[k]:
                    opacity = torch.cat([g.get_opacity for g in v], dim=0)
                    terms[f'reg_{k}_opacity'] = (opacity - 1).pow(2).mean()
                    loss = loss + self.regularizations[k]['lambda_opacity'] * terms[f'reg_{k}_opacity']
            else:
                raise ValueError(f"Invalid representation type: {k}")
        return loss, terms
    
    @torch.no_grad()
    def get_status(self, x: SparseTensor, rep: Dict[str, List]):
        status = {}
        
        for k, v in rep.items():
            if k == 'RF-CP':
                pass
            elif k in ['GS', 'MipGS']:
                xyz = torch.cat([g.get_xyz for g in v], dim=0)
                xyz_base = (x.coords[:, 1:].float() + 0.5) / self.resolution - 0.5
                offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.rep_config[k]['num_gaussians'], -1).reshape(-1, 3)
                _status = {
                    'xyz': xyz,
                    'offset': offset,
                    'scale': torch.cat([g.get_scaling for g in v], dim=0),
                    'opacity': torch.cat([g.get_opacity for g in v], dim=0),
                }

                for kk in list(_status.keys()):
                    _status[kk] = {
                        'mean': _status[kk].mean().item(),
                        'max': _status[kk].max().item(),
                        'min': _status[kk].min().item(),
                    }
                status[k] = _status
            else:
                raise ValueError(f"Invalid representation type: {k}")
            
        return status
    
    def render_batch(self, reps, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = {k: None for k in self.rep_config.keys()}
        for k, v in reps.items():
            for i, representation in enumerate(v):
                render_pack = self.renderers[k].render(representation, extrinsics[i], intrinsics[i])
                if ret[k] is None:
                    ret[k] = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
                for kk, vv in render_pack.items():
                    ret[k][kk].append(vv)
                ret[k]['bg_color'].append(self.renderers[k].bg_color)
            for kk, vv in ret[k].items():
                ret[k][kk] = torch.stack(vv, dim=0) 
        return ret
                    
    def training_losses(self, feats, image, extrinsics, intrinsics, return_aux=False, **kwargs):
        """
        Compute training losses for a single timestep.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        x, mean, logvar = self.backbones['vae'](feats, mem_ratio=self.mem_ratio)
        reps = self.to_representation(x)
        for k, v in self.renderers.items():
            v.rendering_options.resolution = image.shape[-1]
        render_results = self.render_batch(reps, extrinsics, intrinsics)     
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        # concatenate the image for pipeline compatibility
        rec_image = torch.cat([v['rgb'] for v in render_results.values()])
        gt_image = torch.cat([image for v in render_results.values()])
        rec_images = {k: i for k, i in zip(render_results.keys(), torch.chunk(rec_image, len(render_results), dim=0))}
        gt_images = {k: i for k, i in zip(render_results.keys(), torch.chunk(gt_image, len(render_results), dim=0))}
        
        for k in render_results.keys():
            _rec_image = rec_images[k]
            _gt_image = gt_images[k]
            
            if self.loss_type == 'l1':
                terms[k + "_l1"] = l1_loss(_rec_image, _gt_image)
                terms["rec"] = terms["rec"] + terms[k + "_l1"]
            elif self.loss_type == 'l2':
                terms[k + "_l2"] = l2_loss(_rec_image, _gt_image)
                terms["rec"] = terms["rec"] + terms[k + "_l2"]
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")
            if self.lambda_ssim > 0:
                terms[k + "_ssim"] = 1 - ssim(_rec_image, _gt_image)
                terms["rec"] = terms["rec"] + self.lambda_ssim * terms[k + "_ssim"]
            if self.lambda_lpips > 0:
                terms[k + "_lpips"] = lpips(_rec_image, _gt_image)
                terms["rec"] = terms["rec"] + self.lambda_lpips * terms[k + "_lpips"]
            terms["loss"] = terms["loss"] + terms["rec"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lamda_kl * terms["kl"]
        
        reg_loss, reg_terms = self.get_regularization_loss(x, reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        
        # status = self.get_status(x, reps)
        
        if return_aux:
            return terms, reps, {'rec_image': rec_image, 'gt_image': gt_image}       
        return terms, reps

    def encode_decode(self, feats, image, extrinsics, intrinsics, return_aux=False, **kwargs):
        x, mean, logvar = self.backbones['vae'](feats, mem_ratio=self.mem_ratio)
        reps = self.to_representation(x)
        for k, v in self.renderers.items():
            v.rendering_options.resolution = image.shape[-1]
        render_results = self.render_batch(reps, extrinsics, intrinsics)
        rec_image = torch.cat([v['rgb'] for v in render_results.values()])
        gt_image = torch.cat([image for v in render_results.values()])
        if return_aux:
            return reps, {'x': x, 'rec_image': rec_image, 'gt_image': gt_image, 'mean': mean, 'logvar': logvar}
        return reps

    def encode_decode_no_render(self, feats, return_aux=False, **kwargs):
        x, mean, logvar = self.backbones['vae'](feats, mem_ratio=self.mem_ratio)
        reps = self.to_representation(x)
        if return_aux:
            return reps, {'x': x, 'mean': mean, 'logvar': logvar}
        return reps

    @torch.no_grad()
    def snapshot(self, num_samples, batch_size, dataset, verbose=False):
        data_splits = []
        train_dataset = copy.deepcopy(dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        )
        data_splits.append(('train', train_loader))
        if hasattr(dataset, 'set_split'):
            test_dataset = copy.deepcopy(dataset)
            test_dataset.set_split('test-object')
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            )
            data_splits.append(('test', test_loader))

        # inference
        ret_dict = {}
        for split, loader in data_splits:
            gt_images = []
            exts = []
            ints = []
            xs = []
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                data = next(iter(loader))
                args = {k: v[:batch].cuda() for k, v in data.items()}
                x = self.backbones['vae'](args['feats'])[0]
                gt_images.append(args['image'] * args['alpha'][:, None])
                exts.append(args['extrinsics'])
                ints.append(args['intrinsics'])
                xs.append(x)

            gt_images = torch.cat(gt_images, dim=0)
            exts = torch.cat(exts, dim=0)
            ints = torch.cat(ints, dim=0)
            xs = sparse_cat(xs)
            reps = self.to_representation(xs)
            for k, v in self.renderers.items():
                v.rendering_options.bg_color = (0, 0, 0)
                v.rendering_options.resolution = gt_images.shape[-1]
            render_results = self.render_batch(reps, exts, ints)
            for k, v in render_results.items():
                ret_dict.update({f'{split}_rec_{k}_image': {'value': v['rgb'], 'type': 'image'}})

            # render multiview
            for k, v in self.renderers.items():
                v.rendering_options.resolution = 512
            ## Build camera
            yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            yaws = [y + yaws_offset for y in yaws]
            pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

            ## render each view
            miltiview_images_dict = {}
            for yaw, pitch in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(30)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                extrinsics = extrinsics.unsqueeze(0).expand(xs.shape[0], -1, -1)
                intrinsics = intrinsics.unsqueeze(0).expand(xs.shape[0], -1, -1)
                render_results = self.render_batch(reps, extrinsics, intrinsics)
                for k, v in render_results.items():
                    if k not in miltiview_images_dict:
                        miltiview_images_dict[k] = []
                    miltiview_images_dict[k].append(v['rgb'])

            ## Concatenate views
            for k, miltiview_images in miltiview_images_dict.items():
                miltiview_images = torch.cat([
                    torch.cat(miltiview_images[:2], dim=-2),
                    torch.cat(miltiview_images[2:], dim=-2),
                ], dim=-1)
                ret_dict.update({f'{split}_miltiview_{k}_image': {'value': miltiview_images, 'type': 'image'}})

            for k, v in self.renderers.items():
                v.rendering_options.bg_color = 'random'
            
            ret_dict.update({f'{split}_gt_image': {'value': gt_images, 'type': 'image'}})
                            
        return ret_dict
    
    def encode(self, feats, **kwargs):
        return self.backbones['vae'].encode(feats, **kwargs)
    
    def decode(self, latent):
        x = self.backbones['vae'].decode(latent)
        return self.to_representation(x)
    
