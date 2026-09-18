# Converted from configs/swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .swin_tiny_patch4_window7_in1k_pre_upernet_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model.merge(dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150)))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
