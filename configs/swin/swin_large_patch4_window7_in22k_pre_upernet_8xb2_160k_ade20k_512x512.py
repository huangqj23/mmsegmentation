# Converted from configs/swin/swin-large-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .swin_tiny_patch4_window7_in1k_pre_upernet_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa
model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=150),
    auxiliary_head=dict(in_channels=768, num_classes=150)))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
