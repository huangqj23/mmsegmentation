# Converted from configs/sem_fpn/fpn_r101_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .fpn_r50_4xb4_160k_ade20k_512x512 import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_model = model
# ⚠️ 上游原配置中 model 被赋值 2 次；旧式继承只合并最后一次的值，
# 之前的赋值不会进入最终配置。此处如实保留该行为。
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
model = dict(data_preprocessor=data_preprocessor)

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
model = __iv_merge(__base_model, model)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
