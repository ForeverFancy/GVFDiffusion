from typing import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import zero_module, convert_module_to_f16, convert_module_to_f32
import sparse as sp


def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


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


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, hidden_size, in_channels=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x):
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

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.hidden_size:
            embed = torch.cat([embed, torch.zeros(N, self.hidden_size - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class SparseFeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            sp.SparseLinear(hidden_size, int(hidden_size * mlp_ratio)),
            sp.SparseGELU(approximate="tanh"),
            sp.SparseLinear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_mode="full",
        window_size=1024,
        shift_sequence=0,
        shift_window=(0, 0, 0),
        serialize_mode=sp.SerializeMode.Z_ORDER,
        use_checkpoint=False,
        modulated=True,
        use_rope=False,
        use_old_attn_impl=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = sp.SparseMultiHeadAttention(
            hidden_size,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            qkv_bias=True,
            serialize_mode=serialize_mode,
            use_rope=use_rope,
            use_old_attn_impl=use_old_attn_impl,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SparseFeedForward(
            hidden_size,
            mlp_ratio=mlp_ratio,
        )
        self.modulated = modulated
        if modulated:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def _forward(self, x: sp.SparseTensor, c: torch.Tensor):
        if self.modulated:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            h = x.replace(self.norm1(x.feats))
            h = h * (1 + scale_msa) + shift_msa
            h = self.attn(h)
            h = h * gate_msa
            x = x + h
            h = x.replace(self.norm2(x.feats))
            h = h * (1 + scale_mlp) + shift_mlp
            h = self.mlp(h)
            h = h * gate_mlp
            x = x + h
        else:
            h = x.replace(self.norm1(x.feats))
            h = self.attn(h)
            x = x + h
            h = x.replace(self.norm2(x.feats))
            h = self.mlp(h)
            x = x + h
        return x

    def forward(self, x: sp.SparseTensor, c: torch.Tensor = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c, use_reentrant=False)
        else:
            return self._forward(x, c)


class SparseTransformer(nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        model_channels,
        out_channels,
        num_blocks,
        window_size=1024,
        num_heads=None,
        num_head_channels=64,
        mlp_ratio=4,
        attn_mode="swin",
        pe_mode="ape",
        use_fp16=False,
        use_checkpoint=False,
        time_modulated=True,
        use_old_attn_impl=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.time_modulated = time_modulated
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if self.time_modulated:
            self.t_embedder = TimestepEmbedder(model_channels)

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
            
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                modulated=self.time_modulated,
                use_rope=(pe_mode == "rope"),
                use_old_attn_impl=use_old_attn_impl,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        # if use_fp16:
        #     self.convert_to_fp16()

    @property
    def device(self):
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    @property
    def example_inputs(self):
        """
        Return example inputs for model summary.
        """
        return {
            "x": sp.SparseTensor.full((0, 0, 0, 15, 15, 15), (1, self.in_channels), 0).to(self.device),
            "times": torch.zeros((1,), dtype=torch.long).to(self.device),
            "classes": torch.randint(0, self.num_classes, (1,)).to(self.device) if self.num_classes is not None else None,
        }     

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.time_modulated:
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor = None, c: torch.Tensor = None, **kwargs):
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        t_emb = self.t_embedder(t).type(self.dtype) if self.time_modulated else None
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h

