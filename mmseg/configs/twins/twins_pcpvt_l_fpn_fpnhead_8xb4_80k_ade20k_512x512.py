# Converted from configs/twins/twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .twins_pcpvt_s_fpn_fpnhead_8xb4_80k_ade20k_512x512 import *  # noqa: F401,F403

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_large_20220308-37579dc6.pth'  # noqa

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 8, 27, 3])))
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
