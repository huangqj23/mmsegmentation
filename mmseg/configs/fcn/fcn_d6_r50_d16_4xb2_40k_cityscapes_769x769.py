# Converted from configs/fcn/fcn-d6_r50-d16_4xb2-40k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes_769x769 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_40k import *  # noqa: F401,F403
crop_size = (769, 769)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(align_corners=True, dilation=6),
    auxiliary_head=dict(align_corners=True, dilation=6),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
