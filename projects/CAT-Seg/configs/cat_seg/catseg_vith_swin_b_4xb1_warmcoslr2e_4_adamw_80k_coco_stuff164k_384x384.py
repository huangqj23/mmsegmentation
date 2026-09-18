# Converted from projects/CAT-Seg/configs/cat_seg/catseg_vith-swin-b_4xb1-warmcoslr2e-4_adamw-80k_coco-stuff164k_384x384.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .catseg_vitl_swin_b_4xb1_warmcoslr2e_4_adamw_80k_coco_stuff164k_384x384 import *  # noqa: F401,F403
  # noqa

model.merge(dict(
    backbone=dict(
        type='CLIPOVCATSeg',
        clip_pretrained='ViT-H',
        custom_clip_weights='~/CLIP-ViT-H-14-laion2B-s32B-b79K'),
    neck=dict(
        text_guidance_dim=1024,
        appearance_guidance_dim=512,
    )))
