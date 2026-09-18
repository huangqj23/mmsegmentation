# Converted from configs/twins/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.twins_pcpvt_s_fpn import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403
crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))
optim_wrapper.merge(dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=None))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
