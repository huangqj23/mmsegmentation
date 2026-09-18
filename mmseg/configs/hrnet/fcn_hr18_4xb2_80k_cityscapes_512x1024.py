# Converted from configs/hrnet/fcn_hr18_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_hr18 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
crop_size = (512, 1024)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
