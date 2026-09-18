# Converted from configs/pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-512x1024_dark-zurich-1920x1080.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.pspnet_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403

crop_size = (512, 1024)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_dataloader.merge(dict(
    dataset=dict(
        type='DarkZurichDataset',
        data_root='data/dark_zurich/',
        data_prefix=dict(
            img_path='rgb_anon/val/night/GOPR0356',
            seg_map_path='gt/val/night/GOPR0356'),
        pipeline=test_pipeline)))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
