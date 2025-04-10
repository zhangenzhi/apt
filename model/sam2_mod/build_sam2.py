import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    
    if config_file == "sam2_hiera_t.yaml":
        _load_checkpoint(model.image_encoder.trunk.blocks, "./model/sam2_hiera_t.pth")
        _load_checkpoint(model.image_encoder.neck, "./model/sam2_fpn_neck_t.pth")
    elif config_file == "sam2_hiera_b.yaml":
        _load_checkpoint(model.image_encoder.trunk.blocks, "./model/sam2_hiera_b.pth")
        _load_checkpoint(model.image_encoder.neck, "./model/sam2_fpn_neck_b.pth")
    elif config_file == "sam2_hiera_l.yaml":
        _load_checkpoint(model.image_encoder.trunk.blocks, "./model/sam2_hiera_l.pth")
        _load_checkpoint(model.image_encoder.neck, "./model/sam2_fpn_neck_l.pth")

    return model.image_encoder

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")