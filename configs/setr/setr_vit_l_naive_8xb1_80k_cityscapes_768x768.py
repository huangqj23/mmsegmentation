# Converted from configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-768x768.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.models.setr_naive import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes_768x768 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403

from mmengine.model import PretrainedInit
from mmengine.optim import OptimWrapper


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
crop_size = (768, 768)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        drop_rate=0.,
        init_cfg=dict(
            type=PretrainedInit, checkpoint='pretrain/vit_large_p16.pth')),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512))))

optimizer.merge(dict(weight_decay=0.0))
optim_wrapper.merge(dict(
    type=OptimWrapper,
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})))
train_dataloader.merge(dict(batch_size=1))
val_dataloader.merge(dict(batch_size=1))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
