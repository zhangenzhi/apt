import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from typing import Any, Dict, List, Tuple, Type

from model.vit import ImageEncoderViT
from model.vit import LayerNorm2d

def build_sam_vit_b(patch_size=8, image_size=[512, 512], pretrain=True, qdt=False, in_chans=3, use_qdt_pos = False, linear_embed = False):
    return _build_sam_vit(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        patch_size=patch_size,
        image_size=image_size,
        pretrain=pretrain,
        in_chans=in_chans,
        qdt=qdt,
        use_qdt_pos = use_qdt_pos,
        linear_embed = linear_embed
    )

def build_sam_vit_l(patch_size=8, image_size=[512, 512], pretrain=True, qdt=False,):
    return _build_sam_vit(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        patch_size=patch_size,
        image_size=image_size,
        pretrain=pretrain,
        qdt=qdt
    )
    
def build_sam_vit_h(patch_size=8, image_size=[512, 512], pretrain=True, qdt=False,):
    return _build_sam_vit(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        patch_size=patch_size,
        image_size=image_size,
        pretrain=pretrain,
        qdt=qdt,
    )


def _build_sam_vit(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    patch_size,
    image_size,
    pretrain = True,
    in_chans=3,
    qdt=False,
    use_qdt_pos = False,
    linear_embed = False
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = patch_size
    
    image_encoder=ImageEncoderViT(
            in_chans=in_chans,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=False,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            pretrain=pretrain,
            use_abs_pos=False,
            qdt=qdt,
            use_qdt_pos= use_qdt_pos,
            linear_embed= linear_embed,
        )
    
    return image_encoder

class SAM(nn.Module):
    def __init__(self, 
                 image_shape=(512, 512), 
                 patch_size=8,
                 output_dim=1, 
                 pretrain="sam-b"):
        
        super().__init__()
        self.patch_size = patch_size
        if pretrain== "sam-b":
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape)
        elif pretrain== "sam-l":
            self.transformer = build_sam_vit_l(patch_size=self.patch_size, image_size=image_shape)
        elif pretrain=="sam-h":
            self.transformer = build_sam_vit_h(patch_size=self.patch_size, image_size=image_shape)
        else:
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape, pretrain=False)
        
        import math
        upscaling_factor = image_shape[0]// (image_shape[0]/patch_size) if image_shape[0]<=4096 else 4096// (image_shape[0]/patch_size)
        upscaling_factor = int(math.log2(upscaling_factor))
        self.upscale_blocks = nn.ModuleList()
        for i in range(upscaling_factor):
            if i == 0:
                self.upscale_blocks.append(nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=0))
            else:
                self.upscale_blocks.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0))
            self.upscale_blocks.append(LayerNorm2d(64))
            self.upscale_blocks.append(nn.GELU())
        self.mask_header =  nn.Conv2d(64, output_dim, 1)
        self.resize = nn.Upsample((image_shape[0],image_shape[1]))
        
    def forward(self, x):
        # print(x.shape)
        x = self.transformer(x) 
        # print("vit shape:",x.shape)
        for layer in self.upscale_blocks:
            x = layer(x)
        x = self.mask_header(x)
        x = self.resize(x)
        # print("mask shape:",x.shape)
        return x

class SAMQDT(nn.Module):
    def __init__(self, image_shape=(4*32, 4*32), 
                 patch_size=4,
                 in_chans=3,
                 output_dim=1, 
                 pretrain="sam-b",
                 qdt=True,
                 use_qdt_pos = True, 
                 linear_embed = True):
        super().__init__()
        self.patch_size = patch_size
        if pretrain== "sam-b":
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape, 
                                               qdt=qdt, in_chans=patch_size*patch_size*in_chans,use_qdt_pos=use_qdt_pos,linear_embed=linear_embed)
        elif pretrain== "sam-l":
            self.transformer = build_sam_vit_l(patch_size=self.patch_size, image_size=image_shape, qdt=qdt)
        elif pretrain=="sam-h":
             self.transformer = build_sam_vit_h(patch_size=self.patch_size, image_size=image_shape, qdt=qdt)
             
        if not qdt:
            self.mask_header = \
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(128),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(128),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, output_dim, 1)
            )
        else:
            if pretrain== "sam-b":
                self.mask_header = nn.Sequential(nn.Conv2d(256, output_dim, 1))
            elif pretrain== "sam-l":
                self.mask_header = nn.Sequential(nn.Conv2d(256, output_dim, 1))
            elif pretrain== "sam-h":
                self.mask_header = nn.Sequential(nn.Conv2d(256, output_dim, 1))
                
    def forward(self, x, seq_ps=None):
        # print(x.shape)
        x = self.transformer(x, seq_ps=seq_ps) 
        # print("vit shape:",x.shape)
        x = self.mask_header(x)
        # print("mask shape:",x.shape)
        return x

class SAM3D(nn.Module):
    def __init__(self, 
                 image_shape=(512, 512), 
                 patch_size=8,
                 output_dim=1, 
                 pretrain="sam-b"):
        
        super().__init__()
        self.patch_size = patch_size
        if pretrain== "sam-b":
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape)
        elif pretrain== "sam-l":
            self.transformer = build_sam_vit_l(patch_size=self.patch_size, image_size=image_shape)
        elif pretrain=="sam-h":
            self.transformer = build_sam_vit_h(patch_size=self.patch_size, image_size=image_shape)
        else:
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape, pretrain=False)
        
        import math
        upscaling_factor = image_shape[0]// (image_shape[0]/patch_size) if image_shape[0]<=4096 else 4096// (image_shape[0]/patch_size)
        upscaling_factor = int(math.log2(upscaling_factor))
        self.upscale_blocks = nn.ModuleList()
        for i in range(upscaling_factor):
            if i == 0:
                self.upscale_blocks.append(nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=0))
            else:
                self.upscale_blocks.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0))
            self.upscale_blocks.append(LayerNorm2d(64))
            self.upscale_blocks.append(nn.GELU())
        self.mask_header =  nn.Conv2d(64, output_dim, 1)
        self.resize = nn.Upsample((image_shape[0],image_shape[1]))
        
    def forward(self, x):
        # print(x.shape)
        x = self.transformer(x) 
        # print("vit shape:",x.shape)
        for layer in self.upscale_blocks:
            x = layer(x)
        x = self.mask_header(x)
        x = self.resize(x)
        # print("mask shape:",x.shape)
        return x
