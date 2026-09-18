# Converted from configs/cgnet/cgnet_fcn_4xb4-60k_cityscapes-680x680.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.models.cgnet import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import OptimWrapper, PolyLR
from mmengine.runner import IterBasedTrainLoop, TestLoop, ValLoop
from torch.optim import Adam

from mmseg.datasets import LoadAnnotations, PackSegInputs, RandomCrop
from mmseg.datasets.transforms import RandomFlip, Resize


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# optimizer
optimizer = dict(type=Adam, lr=0.001, eps=1e-08, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        by_epoch=False,
        begin=0,
        end=60000)
]
# runtime settings
total_iters = 60000
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=total_iters, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=4000),
    sampler_seed=dict(type=DistSamplerSeedHook))

crop_size = (680, 680)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomResize,
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_dataloader.merge(dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(
    batch_size=1, num_workers=4, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
