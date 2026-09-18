# Converted from configs/swin/swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .swin_base_patch4_window12_in1k_384x384_pre_upernet_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
model.merge(dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
