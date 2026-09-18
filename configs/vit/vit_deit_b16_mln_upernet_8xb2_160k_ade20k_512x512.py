# Converted from configs/vit/vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .vit_vit_b16_mln_upernet_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

model.merge(dict(
    pretrained='pretrain/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(drop_path_rate=0.1),
))
