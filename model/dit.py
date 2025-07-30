from contextlib import contextmanager
import math
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sparse_voxel_diffusion.utils import zero_module
from .attention import MultiHeadAttention
from .sparse_attention import SparseMultiHeadAttention, SerializeMode
from .sparse_voxel_diffusion.elastic_utils import ElasticModule

import sparse as sp


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        B, L, D = x.shape
        x = x.reshape(B*L, D)
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed.reshape(B, L, -1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CanonicalFrameEmbedder(nn.Module):
    """
    Embeds canonical frame index into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, canon_idx):
        """
        Args:
            canon_idx: (B,) tensor of canonical frame indices
            num_frames: total number of frames in sequence
        """
        # Use the same frequency embedding as timestep
        canon_freq = TimestepEmbedder.timestep_embedding(canon_idx, self.frequency_embedding_size)
        canon_emb = self.mlp(canon_freq)
        return canon_emb


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        no_temporal_attn: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.no_temporal_attn = no_temporal_attn
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6) if not no_temporal_attn else nn.Identity()
        self.norm3 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.norm4 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.norm5 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.spatial_self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.temporal_self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        ) if not no_temporal_attn else nn.Identity()
        self.image_cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.static_cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
            self.adaLN_modulation_temporal = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True)
            ) if not no_temporal_attn else nn.Identity()

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, cond_images: torch.Tensor, static_latent: sp.SparseTensor) -> torch.Tensor:
        '''
        x: (B, T, N, C)
        mod: (B, C)
        cond_images: (B, T, L, C)
        static_latent: (B*T, N, C)
        '''
        if self.share_mod:
            if not self.no_temporal_attn:
                shift_msa_spatial, scale_msa_spatial, gate_msa_spatial, shift_msa_temporal, scale_msa_temporal, gate_msa_temporal, shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
            else:
                shift_msa_spatial, scale_msa_spatial, gate_msa_spatial, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa_spatial, scale_msa_spatial, gate_msa_spatial, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            if not self.no_temporal_attn:
                shift_msa_temporal, scale_msa_temporal, gate_msa_temporal = self.adaLN_modulation_temporal(mod).chunk(3, dim=1)
        
        B, T, N, C = x.shape
        # spatial self attention
        h = self.norm1(x)
        h = h * (1 + scale_msa_spatial.unsqueeze(1).unsqueeze(1)) + shift_msa_spatial.unsqueeze(1).unsqueeze(1)
        h = self.spatial_self_attn(h.reshape(B*T, N, C)).reshape(B, T, N, C)
        h = h * gate_msa_spatial.unsqueeze(1).unsqueeze(1)
        x = x + h

        # temporal self attention
        if not self.no_temporal_attn:
            h = self.norm2(x)
            h = h * (1 + scale_msa_temporal.unsqueeze(1).unsqueeze(1)) + shift_msa_temporal.unsqueeze(1).unsqueeze(1)
            h = h.transpose(1, 2).reshape(B*N, T, C)
            h = self.temporal_self_attn(h)
            h = h.reshape(B, N, T, C).transpose(1, 2)
            h = h * gate_msa_temporal.unsqueeze(1).unsqueeze(1)
            x = x + h

        # image cross attention
        h = self.norm3(x)
        h = self.image_cross_attn(h.view(B*T, N, C), cond_images.view(B*T, cond_images.shape[2], C)).view(B, T, N, C)
        x = x + h
        
        # static cross attention
        h = self.norm4(x)
        h = self.static_cross_attn(h.view(B*T, N, C), static_latent.view(B*T, static_latent.shape[2], C)).view(B, T, N, C)
        x = x + h

        # mlp
        h = self.norm5(x)
        h = h * (1 + scale_mlp.unsqueeze(1).unsqueeze(1)) + shift_mlp.unsqueeze(1).unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1).unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, cond_images: torch.Tensor, static_latent: sp.SparseTensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, cond_images, static_latent, use_reentrant=False)
        else:
            return self._forward(x, mod, cond_images, static_latent)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(ElasticModule):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        static_cond_channels: int,
        image_cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 1,
        pe_mode: Literal["ape", "rope", "learnable", "none"] = "learnable",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        no_temporal_attn: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.static_cond_channels = static_cond_channels
        self.image_cond_channels = image_cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.no_temporal_attn = no_temporal_attn

        assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"

        self.t_embedder = TimestepEmbedder(model_channels)
        
        if share_mod:
            # Increase the output size to include canonical frame modulation
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels if not no_temporal_attn else 9 * model_channels, bias=True)  # Modified input size
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif pe_mode == "learnable":
            self.pos_embedder = nn.Parameter(torch.randn(1, resolution, model_channels))

        self.input_layer = nn.Linear(in_channels, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                no_temporal_attn=self.no_temporal_attn,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(model_channels, out_channels)

        self.static_cond_proj = nn.Linear(static_cond_channels, model_channels)
        self.image_cond_proj = nn.Linear(image_cond_channels, model_channels)

        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.static_cond_proj.weight, std=0.02)
        nn.init.normal_(self.image_cond_proj.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @contextmanager
    def with_mem_raio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        blocks = [m for m in self.blocks]
        num_blocks = len(blocks)
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        for i in range(num_blocks):
            blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            blocks[i].use_checkpoint = False
    
    def _forward_with_mem_ratio(self, x: torch.Tensor, t: torch.Tensor, cond_images: torch.Tensor, static_latent: sp.SparseTensor, deformation_position_xyz: torch.Tensor = None, mem_ratio=1.0) -> torch.Tensor:
        with self.with_mem_raio(mem_ratio) as exact_mem_ratio:
            h = self._forward(x, t, cond_images, static_latent, deformation_position_xyz)
        return exact_mem_ratio, h

    def _forward(self, x: torch.Tensor, t: torch.Tensor, cond_images: torch.Tensor, static_latent: sp.SparseTensor, deformation_position_xyz: torch.Tensor = None) -> torch.Tensor:
        '''
        x: (B, T, N, C)
        t: (B,)
        cond_images: (B, T, L, C)
        static_latent: (B, N, C)
        '''
        B, T, N, C = x.shape
        h = self.input_layer(x)
        
        t_emb = self.t_embedder(t)
        
        # Combine timestep and canonical frame embeddings
        combined_emb = t_emb
        
        image_emb = self.image_cond_proj(cond_images)
        static_emb = self.static_cond_proj(static_latent).unsqueeze(1).repeat(1, T, 1, 1)

        if self.share_mod:
            combined_emb = self.adaLN_modulation(combined_emb)
        
        if self.pe_mode == "ape":
            assert deformation_position_xyz is not None, "Deformation position xyz is required for APE mode"
            h = h + self.pos_embedder(deformation_position_xyz).unsqueeze(1).repeat(1, T, 1, 1)
        elif self.pe_mode == "learnable":
            h = h + self.pos_embedder
        
        for block in self.blocks:
            h = block(h, combined_emb, image_emb, static_emb)

        h = self.final_layer(h, combined_emb)
        return h
