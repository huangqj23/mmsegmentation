# Converted from configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_pascal-context-59-480x480.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.deeplabv3plus_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.pascal_context_59 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
crop_size = (480, 480)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=59),
    auxiliary_head=dict(num_classes=59),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320))))
optimizer.merge(dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))
optim_wrapper.merge(dict(type='OptimWrapper', optimizer=optimizer))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
