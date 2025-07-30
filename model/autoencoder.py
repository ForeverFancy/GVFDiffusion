import os
from functools import wraps
import numpy as np
import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint

from einops import rearrange, repeat

from torch_cluster import fps
import pytorch3d.ops

from timm.models.layers import DropPath, trunc_normal_

_FORCE_MEM_EFFICIENT_ATTN = int(os.environ.get('FORCE_MEM_EFFICIENT_ATTN', 0))
print('FORCE_MEM_EFFICIENT_ATTN=', _FORCE_MEM_EFFICIENT_ATTN)
if _FORCE_MEM_EFFICIENT_ATTN:
    from xformers.ops import memory_efficient_attention  # noqa


def timestep_embedding(timesteps, dim, max_period=10000, dtype=None):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if dtype is None:
        dtype = torch.float32
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device, dtype=dtype)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm_context = nn.LayerNorm(context_dim, eps=1e-6, elementwise_affine=False) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0, enable_flash_attn = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.enable_flash_attn = enable_flash_attn

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        B, N, C = x.shape
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        if self.enable_flash_attn:
            from flash_attn import flash_attn_func
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, k, v))

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = x.reshape(B, N, C)
            return self.drop_path(self.to_out(x))
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
            if _FORCE_MEM_EFFICIENT_ATTN:
                out = memory_efficient_attention(q, k, v)
            else:
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h = h)
                    sim.masked_fill_(~mask, max_neg_value)

                # attention, what we cannot get enough of
                attn = sim.softmax(dim = -1)

                out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
            return self.drop_path(self.to_out(out))

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48):
        super().__init__()

        assert hidden_dim % 3 == 0
        self.embedding_dim = hidden_dim // 3 // 2

        omega = np.arange(self.embedding_dim, dtype=np.float64)
        omega /= self.embedding_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        self.register_buffer('omega', torch.from_numpy(omega))

    def forward(self, input):
        # input: B x T x N x 3 or B x N x 3
        input_type = input.dtype
        if len(input.shape) == 3:
            B, N, D = input.shape
            pos = input.reshape(B*N, D)  # (B*N, 3)
            
            # Split into x,y,z components
            emb_x = einsum('m,d->md', pos[:, 0], self.omega)  # (B*N, D/3)
            emb_y = einsum('m,d->md', pos[:, 1], self.omega)  # (B*N, D/3)
            emb_z = einsum('m,d->md', pos[:, 2], self.omega)  # (B*N, D/3)
            
            # Apply sin/cos to each component
            emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=1)  # (B*N, D/3)
            emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=1)  # (B*N, D/3)
            emb_z = torch.cat([torch.sin(emb_z), torch.cos(emb_z)], dim=1)  # (B*N, D/3)
            
            # Concatenate all components
            embed = torch.cat([emb_x, emb_y, emb_z], dim=1)  # (B*N, D)
            embed = embed.reshape(B, N, -1)
            
        elif len(input.shape) == 4:
            B, T, N, D = input.shape
            pos = input.reshape(B*T*N, D)  # (B*T*N, 3)
            
            # Split into x,y,z components
            emb_x = einsum('m,d->md', pos[:, 0], self.omega)  # (B*T*N, D/3)
            emb_y = einsum('m,d->md', pos[:, 1], self.omega)  # (B*T*N, D/3)
            emb_z = einsum('m,d->md', pos[:, 2], self.omega)  # (B*T*N, D/3)
            
            # Apply sin/cos to each component
            emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=1)  # (B*T*N, D/3)
            emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=1)  # (B*T*N, D/3)
            emb_z = torch.cat([torch.sin(emb_z), torch.cos(emb_z)], dim=1)  # (B*T*N, D/3)
            
            # Concatenate all components
            embed = torch.cat([emb_x, emb_y, emb_z], dim=1)  # (B*T*N, D)
            embed = embed.reshape(B, T, N, -1)
            
        return embed.to(input_type)


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class GSKLTemporalVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        input_dim=3,
        gs_dim=14,
        output_dim=10,
        num_inputs=8192,
        num_latents=1024,
        latent_dim=128,
        heads=8,
        dim_head=-1,
        weight_tie_layers=False,
        decoder_ff=False,
        enable_flash_attn=False,
        num_timesteps=24,
        chunk_size=8192,
        knn_k=8,
        beta=7.0,
    ):
        super().__init__()

        self.depth = depth

        if dim_head == -1:
            dim_head = dim // heads

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.interpolation_func = self.compute_delta_interp
        self.knn_k = knn_k
        self.beta = beta
        print(f"Model interpolation using: knn_k={self.knn_k}, beta={self.beta}")

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = heads, dim_head = dim_head, enable_flash_attn=enable_flash_attn), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.input_embedding = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim, elementwise_affine=False))
        self.gs_embedding = nn.Sequential(nn.Linear(gs_dim, dim), nn.LayerNorm(dim, elementwise_affine=False))
        self.position_encoding = nn.Sequential(PointEmbed(hidden_dim=dim), nn.LayerNorm(dim, elementwise_affine=False))

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1, enable_flash_attn=enable_flash_attn))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = heads, dim_head = dim_head, enable_flash_attn=enable_flash_attn), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)
        
        # Add flag for gradient checkpointing
        self.use_checkpoint = True  # You can make this a parameter if needed
        self.chunk_size = chunk_size
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        self.to_outputs = zero_module(self.to_outputs)
    
    def freeze_encoder(self):
        for param in self.cross_attend_blocks.parameters():
            param.requires_grad = False
        for param in self.input_embedding.parameters():
            param.requires_grad = False
        for param in self.position_encoding.parameters():
            param.requires_grad = False
        for param in self.mean_fc.parameters():
            param.requires_grad = False
        for param in self.logvar_fc.parameters():
            param.requires_grad = False  

    def compute_delta_interp(self, static_gs, micro_static_pc, micro_moving_pc, knn_k=8, beta=7.0, adaptive_radius=True):
        """
        Compute interpolation loss between predicted and KNN-estimated deltas
        Args:
            static_gs: static gs [B, N_gs, 3]
            micro_static_pc: Static point cloud [B, N_points, 3]
            micro_moving_pc: Moving point cloud [B, T, N_points, 3]
            knn_k: Number of nearest neighbors for interpolation
            adaptive_radius: If True, use adaptive radius-based filtering
        Returns:
            estimated_deltas: Estimated deltas [B, T, N_points, 3]
        """
        B = static_gs.shape[0]
        T = micro_moving_pc.shape[1]

        with torch.no_grad():
            # Find K nearest neighbors and distances
            knn_dists, knn_idx, _ = pytorch3d.ops.knn_points(
                static_gs,
                micro_static_pc,
                K=knn_k
            )

            adaptive_radii = knn_dists.mean(dim=-1).sqrt() + 1e-6  # [B, N_gs]
            
            # Apply adaptive radius-based filtering
            if adaptive_radius:
                radius_mask = knn_dists <= adaptive_radii[..., None] ** 2
                weights = torch.exp(-beta * knn_dists / adaptive_radii[..., None] ** 2)
                weights = weights * radius_mask.float()
            else:
                weights = torch.exp(-beta * knn_dists)

            # Normalize weights
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Gather and weight point movements
            batch_idx = torch.arange(B).view(-1, 1, 1, 1).expand(-1, static_gs.shape[1], knn_k, T).to(micro_static_pc.device)
            time_idx = torch.arange(T).view(1, 1, 1, -1).expand(B, static_gs.shape[1], knn_k, -1).to(micro_static_pc.device)
            knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, T)
            
            # Get movements of neighboring points
            neighbor_positions = micro_moving_pc[batch_idx, time_idx, knn_idx_expanded]  # [B, N_gs, K, T, 3]
            neighbor_movements = neighbor_positions - micro_static_pc[batch_idx, knn_idx_expanded]
            
            # Apply weights to compute interpolated movement
            weights = weights.unsqueeze(-1).unsqueeze(-2)  # [B, N_gs, K, 1, 1]
            interpolated_movements = (weights * neighbor_movements).sum(dim=2)  # [B, N_gs, T, 3]
            estimated_deltas = interpolated_movements.transpose(1, 2)  # [B, T, N_gs, 3]

        return estimated_deltas

    def encode(self, static_pc, delta_pc, static_gs_list):
        '''
        static_pc: B x N x 3
        delta_pc: B x T x N x 3
        static_gs_list: [N_gs x 14]
        '''
        B, N_pc, D = static_pc.shape
        Batch, T = delta_pc.shape[:2]
        
        # First handle static_gs_list
        gs_lengths = [gs.shape[0] for gs in static_gs_list]
        
        # Create batch indices for gaussians
        gs_batch = []
        for i in range(B):
            gs_batch.extend([i] * gs_lengths[i])
        gs_batch = torch.tensor(gs_batch, device=static_pc.device)
        
        # Stack all gaussians into one tensor
        stacked_gs = torch.cat([static_gs_list[i][:, :3] for i in range(len(static_gs_list))], dim=0)
        
        # Calculate ratios for each instance based on number of gaussians
        gs_ratios = torch.tensor([(self.num_latents / gs_len) for gs_len in gs_lengths], device=static_pc.device)
        gs_idx = fps(stacked_gs, gs_batch, ratio=gs_ratios)
        input_static_gs = stacked_gs[gs_idx].reshape(B, self.num_latents, 3)
        sampled_static_gs = torch.cat([static_gs_list[i] for i in range(len(static_gs_list))], dim=0)[gs_idx].reshape(B, self.num_latents, 14)

        moving_pc = delta_pc + static_pc.unsqueeze(1).repeat(1, T, 1, 1)
        estimated_gs_deltas = self.interpolation_func(input_static_gs, static_pc, moving_pc, knn_k=self.knn_k, beta=self.beta) # [B, T, num_inputs, 3]

        sampled_delta_pc_embeddings = self.input_embedding(estimated_gs_deltas) + self.position_encoding(input_static_gs).unsqueeze(1).repeat(1, T, 1, 1)
        sampled_delta_pc_embeddings = sampled_delta_pc_embeddings.view(Batch*T, self.num_latents, self.dim)

        pc_embeddings = self.input_embedding(delta_pc) + self.position_encoding(static_pc).unsqueeze(1).repeat(1, T, 1, 1)
        pc_embeddings = pc_embeddings.view(Batch*T, self.num_inputs, self.dim)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_delta_pc_embeddings, context = pc_embeddings, mask = None) + sampled_delta_pc_embeddings
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x, posterior, sampled_static_gs
    
    def process_chunk(self, queries_chunk, x):
        """Helper function to process a single chunk with gradient checkpointing"""
        B = queries_chunk.shape[0]
        
        # Move repeat operation outside checkpoint to save memory
        queries_chunk = queries_chunk.unsqueeze(1).repeat(1, self.num_timesteps, 1, 1)
        
        def chunk_forward(q, context):
            q_embed = self.gs_embedding(q) + self.position_encoding(q[..., :3])
            q_embed = q_embed.view(B*self.num_timesteps, -1, self.dim)
            latents = self.decoder_cross_attn(q_embed, context=context)
            
            if exists(self.decoder_ff):
                latents = latents + self.decoder_ff(latents)
                
            return self.to_outputs(latents)
            
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                chunk_forward,
                queries_chunk,
                x,
                use_reentrant=False
            )
        else:
            return chunk_forward(queries_chunk, x)

    def decode(self, x, queries):
        '''
        x: (B x T) x L x D
        queries: B x Q x 14
        '''
        B, num_static_gs = queries.shape[:2]
        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Process in chunks if number of queries is large
        chunk_size = self.chunk_size
        if num_static_gs > chunk_size:
            outputs = []
            for chunk_start in range(0, num_static_gs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_static_gs)
                queries_chunk = queries[:, chunk_start:chunk_end]
                
                # Process chunk with gradient checkpointing
                chunk_output = self.process_chunk(queries_chunk, x)
                outputs.append(chunk_output)
            
            # Concatenate chunks
            output = torch.cat(outputs, dim=1)
        else:
            # Process entire input with gradient checkpointing
            output = self.process_chunk(queries, x)
            
        return output.reshape(B, self.num_timesteps, num_static_gs, -1)

    def pad_static_gs(self, static_gs):
        # pad static_gs to the same length as the longest static_gs, and return mask
        max_len = max([static_gs[i].shape[0] for i in range(len(static_gs))])
        padding = torch.zeros((1, static_gs[0].shape[1])).to(static_gs[0].device)
        padding[0, 10] = 1.0  # Set the 11th element (index 10) to 1
        padded_static_gs = [torch.cat([static_gs[i], padding.repeat(max_len - static_gs[i].shape[0], 1)], dim=0) for i in range(len(static_gs))]
        padded_static_gs = torch.stack(padded_static_gs, dim=0)
        idx = [static_gs[i].shape[0] for i in range(len(static_gs))]
        return padded_static_gs, idx

    def forward(self, static_gs, static_pc, delta_pc):
        kl, x, posterior, sampled_static_pc = self.encode(static_pc, delta_pc, static_gs)

        padded_static_gs, idx = self.pad_static_gs(static_gs)
        o = self.decode(x, padded_static_gs).squeeze(-1)

        return {'logits': o, 'kl': kl, 'posterior': posterior}
