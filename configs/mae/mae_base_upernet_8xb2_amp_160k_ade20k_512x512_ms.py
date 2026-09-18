# Converted from configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512-ms.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .mae_base_upernet_8xb2_amp_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile

from mmseg.datasets import LoadAnnotations, PackSegInputs
from mmseg.datasets.transforms import Resize


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

test_pipeline = [
    dict(type=LoadImageFromFile),
    # TODO: Refactor 'MultiScaleFlipAug' which supports
    # `min_size` feature in `Resize` class
    # img_ratios is [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # original image scale is (2048, 512)
    dict(type=Resize, scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations, reduce_zero_label=True),
    dict(type=PackSegInputs)
]
val_dataloader.merge(dict(batch_size=1, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
