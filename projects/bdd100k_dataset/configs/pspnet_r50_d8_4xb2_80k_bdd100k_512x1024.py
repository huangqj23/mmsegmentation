# Converted from projects/bdd100k_dataset/configs/pspnet_r50-d8_4xb2-80k_bdd100k-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from mmseg.configs._base_.models.pspnet_r50_d8 import *  # noqa: F401,F403
    from ._base_.datasets.bdd100k import *  # noqa: F401,F403
    from mmseg.configs._base_.default_runtime import *  # noqa: F401,F403
    from mmseg.configs._base_.schedules.schedule_80k import *  # noqa: F401,F403
custom_imports = dict(
    imports=['projects.bdd100k_dataset.mmseg.datasets.bdd100k'])
crop_size = (512, 1024)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
