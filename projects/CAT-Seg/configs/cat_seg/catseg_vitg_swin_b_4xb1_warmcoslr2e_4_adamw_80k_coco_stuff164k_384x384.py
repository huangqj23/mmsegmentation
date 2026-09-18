# Converted from projects/CAT-Seg/configs/cat_seg/catseg_vitg-swin-b_4xb1-warmcoslr2e-4_adamw-80k_coco-stuff164k_384x384.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .catseg_vitl_swin_b_4xb1_warmcoslr2e_4_adamw_80k_coco_stuff164k_384x384 import *  # noqa: F401,F403
  # noqa

model.merge(dict(
    backbone=dict(
        type='CLIPOVCATSeg',
        clip_pretrained='ViT-G',
        custom_clip_weights='~/CLIP-ViT-bigG-14-laion2B-39B-b160k'),
    neck=dict(
        text_guidance_dim=1280,
        appearance_guidance_dim=512,
    )))
