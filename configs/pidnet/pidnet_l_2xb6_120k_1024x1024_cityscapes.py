# Converted from configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pidnet_s_2xb6_120k_1024x1024_cityscapes import *  # noqa: F401,F403
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa
model.merge(dict(
    backbone=dict(
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(in_channels=256, channels=256)))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
