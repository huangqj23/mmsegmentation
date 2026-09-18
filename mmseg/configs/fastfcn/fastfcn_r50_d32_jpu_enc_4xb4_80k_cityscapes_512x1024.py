# Converted from configs/fastfcn/fastfcn_r50-d32_jpu_enc_4xb4-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
# model settings
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .fastfcn_r50_d32_jpu_enc_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
train_dataloader.merge(dict(batch_size=4, num_workers=4))
val_dataloader.merge(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
