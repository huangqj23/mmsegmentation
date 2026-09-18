# Converted from configs/twins/twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .twins_pcpvt_s_uperhead_8xb4_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_base_20220308-0621964c.pth'  # noqa

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),
        depths=[3, 4, 18, 3],
        drop_path_rate=0.3)))

train_dataloader.merge(dict(batch_size=2, num_workers=2))
val_dataloader.merge(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
