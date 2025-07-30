from contextlib import contextmanager
from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import zero_module, convert_module_to_f16, convert_module_to_f32
import sparse as sp
from .elastic_utils import ElasticModule
from .sparse_transformer import AbsolutePositionEmbedder, SparseTransformerBlock, block_attn_config


class SparseTransformerVAE(ElasticModule):
    def __init__(
        self,
        resolution,
        in_channels,
        model_channels,
        out_channels,
        latent_channels,
        num_blocks,
        window_size=1024,
        num_heads=None,
        num_head_channels=64,
        mlp_ratio=4,
        attn_mode="swin",
        pe_mode="ape",
        use_fp16=False,
        use_checkpoint=False,
        use_old_attn_impl=True,
        norm_output=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.norm_output = norm_output
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.encoder = nn.ModuleList([
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
                modulated=False,
                use_rope=(pe_mode == "rope"),
                use_old_attn_impl=use_old_attn_impl,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])
        self.to_latent = sp.SparseLinear(model_channels, 2 * latent_channels)

        self.from_latent = sp.SparseLinear(latent_channels, model_channels)
        self.decoder = nn.ModuleList([
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
                modulated=False,
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
        self.encoder.apply(convert_module_to_f16)
        self.decoder.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.encoder.apply(convert_module_to_f32)
        self.decoder.apply(convert_module_to_f32)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out to_latent layer:
        nn.init.constant_(self.to_latent.weight, 0)
        nn.init.constant_(self.to_latent.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    
    def freeze_encoder(self):
        for block in self.encoder:
            block.requires_grad_(False)

    def encode(self, x: sp.SparseTensor, sample_posterior=True, return_raw=False):
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        for block in self.encoder:
            h = block(h)
        h = h.type(x.dtype)
        if self.norm_output:
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
            
    def decode(self, latent: sp.SparseTensor):
        h = self.from_latent(latent)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(latent.coords[:, 1:])
        h = h.type(self.dtype)
        for block in self.decoder:
            h = block(h)
        h = h.type(latent.dtype)
        if self.norm_output:
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h
    
    def _get_input_size(self, x: sp.SparseTensor, t: torch.Tensor = None, c: torch.Tensor = None):
        return x.feats.shape[0]

    @contextmanager
    def with_mem_raio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        blocks = [m for m in self.encoder] + [m for m in self.decoder]
        num_blocks = len(blocks)
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        for i in range(num_blocks):
            blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            blocks[i].use_checkpoint = False
    
    def _forward_with_mem_ratio(self, x: sp.SparseTensor, t: torch.Tensor = None, c: torch.Tensor = None, mem_ratio=1.0):
        with self.with_mem_raio(mem_ratio) as exact_mem_ratio:
            latent, mean, logvar = self.encode(x, sample_posterior=True, return_raw=True)
            out = self.decode(latent)
        return exact_mem_ratio, (out, mean, logvar)

