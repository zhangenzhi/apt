import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from typing import Any, Dict, List, Tuple, Type

from model.vit import ImageEncoderViT
from model.vit import LayerNorm2d

def build_sam_vit_b(patch_size=8, image_size=[512, 512], pretrain=True, qdt=False,):
    return _build_sam_vit(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        patch_size=patch_size,
        image_size=image_size,
        pretrain=pretrain,
        qdt=qdt,
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
    pretrain =True,
    qdt=False,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = patch_size
    tokens = image_size[0]//patch_size * image_size[1]//patch_size
    
    image_encoder=ImageEncoderViT(
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
            qdt=qdt,
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
                 output_dim=1, 
                 pretrain="sam-b",
                 qdt=False):
        super().__init__()
        self.patch_size = patch_size
        if pretrain== "sam-b":
            self.transformer = build_sam_vit_b(patch_size=self.patch_size, image_size=image_shape, qdt=qdt)
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
                
    def forward(self, x):
        # print(x.shape)
        x = self.transformer(x) 
        # print("vit shape:",x.shape)
        x = self.mask_header(x)
        # print("mask shape:",x.shape)
        return x
             
# class MaskDecoder(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         transformer: nn.Module,
#         num_multimask_outputs: int = 3,
#         activation: Type[nn.Module] = nn.GELU,
#         iou_head_depth: int = 3,
#         iou_head_hidden_dim: int = 256,
#     ) -> None:
#         """
#         Predicts masks given an image and prompt embeddings, using a
#         transformer architecture.

#         Arguments:
#           transformer_dim (int): the channel dimension of the transformer
#           transformer (nn.Module): the transformer used to predict masks
#           num_multimask_outputs (int): the number of masks to predict
#             when disambiguating masks
#           activation (nn.Module): the type of activation to use when
#             upscaling masks
#           iou_head_depth (int): the depth of the MLP used to predict
#             mask quality
#           iou_head_hidden_dim (int): the hidden dimension of the MLP
#             used to predict mask quality
#         """
#         super().__init__()
#         self.transformer_dim = transformer_dim
#         self.transformer = transformer

#         self.num_multimask_outputs = num_multimask_outputs

#         self.iou_token = nn.Embedding(1, transformer_dim)
#         self.num_mask_tokens = num_multimask_outputs + 1
#         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
#             activation(),
#         )
#         self.output_hypernetworks_mlps = nn.ModuleList(
#             [
#                 MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
#                 for i in range(self.num_mask_tokens)
#             ]
#         )

#         self.iou_prediction_head = MLP(
#             transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
#         )

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         # sparse_prompt_embeddings: torch.Tensor,
#         # dense_prompt_embeddings: torch.Tensor,
#         multimask_output: bool,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Predict masks given image and prompt embeddings.

#         Arguments:
#           image_embeddings (torch.Tensor): the embeddings from the image encoder
#           image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
#           sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
#           dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
#           multimask_output (bool): Whether to return multiple masks or a single
#             mask.

#         Returns:
#           torch.Tensor: batched predicted masks
#           torch.Tensor: batched predictions of mask quality
#         """
#         masks, iou_pred = self.predict_masks(
#             image_embeddings=image_embeddings,
#             image_pe=image_pe,
#             # sparse_prompt_embeddings=sparse_prompt_embeddings,
#             # dense_prompt_embeddings=dense_prompt_embeddings,
#         )

#         # Select the correct mask or masks for output
#         if multimask_output:
#             mask_slice = slice(1, None)
#         else:
#             mask_slice = slice(0, 1)
#         masks = masks[:, mask_slice, :, :]
#         iou_pred = iou_pred[:, mask_slice]

#         # Prepare output
#         return masks, iou_pred

#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         # sparse_prompt_embeddings: torch.Tensor,
#         # dense_prompt_embeddings: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""
#         # Concatenate output tokens
#         output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
#         # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
#         # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

#         # Expand per-image data in batch direction to be per-mask
#         src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
#         # src = src + dense_prompt_embeddings
#         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
#         b, c, h, w = src.shape

#         # Run the transformer
#         hs, src = self.transformer(src, pos_src, tokens)
#         iou_token_out = hs[:, 0, :]
#         mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

#         # Upscale mask embeddings and predict masks using the mask tokens
#         src = src.transpose(1, 2).view(b, c, h, w)
#         upscaled_embedding = self.output_upscaling(src)
#         hyper_in_list: List[torch.Tensor] = []
#         for i in range(self.num_mask_tokens):
#             hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
#         hyper_in = torch.stack(hyper_in_list, dim=1)
#         b, c, h, w = upscaled_embedding.shape
#         masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

#         # Generate mask quality predictions
#         iou_pred = self.iou_prediction_head(iou_token_out)

#         return masks, iou_pred

# # Lightly adapted from
# # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
    # def __init__(
    #     self,
    #     input_dim: int,
    #     hidden_dim: int,
    #     output_dim: int,
    #     num_layers: int,
    #     sigmoid_output: bool = False,
    # ) -> None:
    #     super().__init__()
    #     self.num_layers = num_layers
    #     h = [hidden_dim] * (num_layers - 1)
    #     self.layers = nn.ModuleList(
    #         nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
    #     )
    #     self.sigmoid_output = sigmoid_output

    # def forward(self, x):
    #     for i, layer in enumerate(self.layers):
    #         x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    #     if self.sigmoid_output:
    #         x = F.sigmoid(x)
    #     return x