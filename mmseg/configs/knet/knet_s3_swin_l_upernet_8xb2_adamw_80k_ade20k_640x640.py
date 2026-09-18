# Converted from configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .knet_s3_swin_t_upernet_8xb2_adamw_80k_ade20k_512x512 import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
__base_train_dataloader = train_dataloader
__base_val_dataloader = val_dataloader

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'  # noqa
# model settings
crop_size = (640, 640)
data_preprocessor.merge(dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint_file,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.4,
        patch_norm=True),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768)))

crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
# ⚠️ 上游原配置中 train_dataloader 被赋值 2 次；旧式继承只合并最后一次的值，
# 之前的赋值不会进入最终配置。此处如实保留该行为。
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# ⚠️ 上游原配置中 val_dataloader 被赋值 2 次；旧式继承只合并最后一次的值，
# 之前的赋值不会进入最终配置。此处如实保留该行为。
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# ⚠️ 上游原配置中 test_dataloader 被赋值 2 次；旧式继承只合并最后一次的值，
# 之前的赋值不会进入最终配置。此处如实保留该行为。
test_dataloader = val_dataloader
# In K-Net implementation we use batch size 2 per GPU as default
train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
train_dataloader = __iv_merge(__base_train_dataloader, train_dataloader)
val_dataloader = __iv_merge(__base_val_dataloader, val_dataloader)
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
