# Converted from configs/swin/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .swin_base_patch4_window7_in1k_pre_upernet_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'  # noqa
model.merge(dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
