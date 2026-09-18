# Converted from configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .mask2former_swin_b_in1k_384x384_pre_8xb2_160k_ade20k_640x640 import *  # noqa: F401,F403
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa

model.merge(dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_queries=100, in_channels=[192, 384, 768, 1536])))
if isinstance(pretrained, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    pretrained.pop('_delete_', None)
