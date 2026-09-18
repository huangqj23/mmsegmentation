# Converted from configs/unet/unet-s5-d16_fcn_4xb4-40k_stare-128x128.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_unet_s5_d16 import *  # noqa: F401,F403
    from .._base_.datasets.stare import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_40k import *  # noqa: F401,F403
crop_size = (128, 128)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85))))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
