# Converted from configs/san/san-vit-b16_coco-stuff164k-640x640.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.models.san_vit_b16 import *  # noqa: F401,F403
    from .._base_.datasets.coco_stuff164k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomChoiceResize
from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper, PolyLR
from mmengine.runner import IterBasedTrainLoop
from torch.optim import AdamW

from mmseg.datasets import (LoadAnnotations, PackSegInputs,
                            PhotoMetricDistortion, RandomCrop,
                            ResizeShortestEdge)
from mmseg.datasets.transforms import RandomFlip


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
crop_size = (640, 640)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=1.0),
    dict(type=PhotoMetricDistortion),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackSegInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=ResizeShortestEdge, scale=crop_size, max_size=2560),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]

# By default, models are trained on 4 GPUs with 8 images per GPU
train_dataloader.merge(dict(batch_size=8, dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(batch_size=1, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  # noqa
data_preprocessor.merge(dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32)))
model.merge(dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name='coco-stuff164k'),
    decode_head=dict(num_classes=171)))

# training schedule for 60k
train_cfg.merge(dict(
    type=IterBasedTrainLoop,
    max_iters=60000,
    val_interval=500,
    val_begin=55000))
default_hooks.merge(dict(
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=10000,
        save_best='mIoU')))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper.merge(dict(
    _delete_=True,
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2)))

param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
