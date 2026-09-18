# Converted from configs/twins/twins_svt-l_uperhead_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .twins_svt_s_uperhead_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_large_20220308-fb5936f3.pth'  # noqa

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512)))
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
