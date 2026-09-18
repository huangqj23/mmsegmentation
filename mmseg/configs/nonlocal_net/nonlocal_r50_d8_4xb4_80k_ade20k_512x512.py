# Converted from configs/nonlocal_net/nonlocal_r50-d8_4xb4-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.nonlocal_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150)))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
