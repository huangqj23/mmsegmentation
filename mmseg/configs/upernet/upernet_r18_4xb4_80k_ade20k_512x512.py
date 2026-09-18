# Converted from configs/upernet/upernet_r18_4xb4-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.upernet_r50 import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150)))
