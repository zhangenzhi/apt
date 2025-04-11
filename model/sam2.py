import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append("./")

from model.vit import LayerNorm2d
from model.sam2_mod.build_sam2 import build_sam2

class SAM2(nn.Module):
    def __init__(self, 
                 image_shape=(512, 512), 
                 output_dim=1, 
                 pretrain="sam2-t"):
        
        super().__init__()
        
        self.num_feature_levels = 3
        if pretrain == "sam2-t":
            cfg_path = "sam2_hiera_t.yaml"
            ckpt_path = "./model/sam2_hiera_t.pth"
        elif pretrain == "sam2-s":
            cfg_path = "sam2_hiera_s.yaml"
            ckpt_path = "./model/sam2_hiera_s.pth"
        elif pretrain == "sam2-b":
            cfg_path = "sam2_hiera_b.yaml"
            ckpt_path = "./model/sam2_hiera_b.pth"
        elif pretrain == "sam2-l":
            cfg_path = "sam2_hiera_l.yaml"
            ckpt_path = "./model/sam2_hiera_l.pth"
        else:
            raise "No such pretrained weights"

        self.encoder = build_sam2(config_file=cfg_path, ckpt_path=ckpt_path)
        self._bb_feat_sizes = [
            (image_shape[0]//4, image_shape[0]//4),
            (image_shape[0]//8, image_shape[0]//8),
            (image_shape[0]//16, image_shape[0]//16),
        ]
        self.mask_decoder = \
        nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 1)
        )
            
    def prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
    
    def forward(self, x):
        batch_size,_,_,_ = x.shape
        x = self.encoder(x)
        _, vision_feats, _, _ = self.prepare_backbone_features(x)
        # print(vision_feats[0].shape)
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        x = self.mask_decoder(feats[-1])
        
        return x

class SAM2QDT(nn.Module):
    def __init__(self, image_shape=(4*32, 4*32), 
                 patch_size=4,
                 in_chans=3,
                 output_dim=1, 
                 pretrain="sam-b",
                 qdt=False):
        super().__init__()
        self.patch_size = patch_size
        self.num_feature_levels = 3
        if pretrain == "sam2-t":
            cfg_path = "sam2_hiera_t.yaml"
            ckpt_path = "./model/sam2_hiera_t.pth"
        elif pretrain == "sam2-s":
            cfg_path = "sam2_hiera_s.yaml"
            ckpt_path = "./model/sam2_hiera_s.pth"
        elif pretrain == "sam2-b":
            cfg_path = "sam2_hiera_b.yaml"
            ckpt_path = "./model/sam2_hiera_b.pth"
        elif pretrain == "sam2-l":
            cfg_path = "sam2_hiera_l.yaml"
            ckpt_path = "./model/sam2_hiera_l.pth"
        else:
            raise "No such pretrained weights"

        self.encoder = build_sam2(config_file=cfg_path, ckpt_path=ckpt_path)
        self._bb_feat_sizes = [
            (image_shape[0]//4, image_shape[0]//4),
            (image_shape[0]//8, image_shape[0]//8),
            (image_shape[0]//16, image_shape[0]//16),
        ]
        
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
                
    def forward(self, x):
        # print(x.shape)
        x = self.transformer(x) 
        # print("vit shape:",x.shape)
        x = self.mask_header(x)
        # print("mask shape:",x.shape)
        return x
    
if __name__ == "__main__":
    model = SAM2QDT(image_shape=(256, 256), pretrain="sam2-t")
    batch_size = 4
    x = torch.rand(size=(batch_size, 1, 256, 256))
    embs = model(x)
    print(embs.shape)
    
        
        