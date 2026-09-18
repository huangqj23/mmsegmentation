# Converted from configs/hrnet/fcn_hr18_4xb4-80k_isaid-896x896.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_hr18 import *  # noqa: F401,F403
    from .._base_.datasets.isaid import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
crop_size = (896, 896)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=16)))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
