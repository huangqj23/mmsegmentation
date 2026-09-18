# Converted from projects/medical/2d_image/histopathology/consep/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_consep-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .consep_512x512 import *  # noqa: F401,F403
    from mmseg.configs._base_.models.fcn_unet_s5_d16 import *  # noqa: F401,F403
    from mmseg.configs._base_.default_runtime import *  # noqa: F401,F403
    from mmseg.configs._base_.schedules.schedule_20k import *  # noqa: F401,F403

def __iv_scope(cfg, scope, has_scope=True):
    """复刻旧式跨库继承（scope::path）时 mmengine 给 base 变量加 _scope_ 的逻辑：
    每条路径上最外层带 type 的 dict 加 _scope_。"""
    if isinstance(cfg, dict):
        if has_scope and 'type' in cfg:
            has_scope = False
            if cfg.get('_scope_') is None:
                cfg['_scope_'] = scope
        for k in list(cfg):
            cfg[k] = __iv_scope(cfg[k], scope, has_scope)
    elif isinstance(cfg, list):
        cfg = [__iv_scope(v, scope, has_scope) for v in cfg]
    elif isinstance(cfg, tuple):
        cfg = tuple(__iv_scope(v, scope, has_scope) for v in cfg)
    return cfg

data_preprocessor = __iv_scope(data_preprocessor, 'mmseg')
model = __iv_scope(model, 'mmseg')
norm_cfg = __iv_scope(norm_cfg, 'mmseg')
default_scope = __iv_scope(default_scope, 'mmseg')
env_cfg = __iv_scope(env_cfg, 'mmseg')
load_from = __iv_scope(load_from, 'mmseg')
log_level = __iv_scope(log_level, 'mmseg')
log_processor = __iv_scope(log_processor, 'mmseg')
resume = __iv_scope(resume, 'mmseg')
tta_model = __iv_scope(tta_model, 'mmseg')
vis_backends = __iv_scope(vis_backends, 'mmseg')
visualizer = __iv_scope(visualizer, 'mmseg')
default_hooks = __iv_scope(default_hooks, 'mmseg')
optim_wrapper = __iv_scope(optim_wrapper, 'mmseg')
optimizer = __iv_scope(optimizer, 'mmseg')
param_scheduler = __iv_scope(param_scheduler, 'mmseg')
test_cfg = __iv_scope(test_cfg, 'mmseg')
train_cfg = __iv_scope(train_cfg, 'mmseg')
val_cfg = __iv_scope(val_cfg, 'mmseg')


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_data_preprocessor = data_preprocessor
__base_model = model
__base_optim_wrapper = optim_wrapper
__base_optimizer = optimizer
__base_visualizer = visualizer
custom_imports = dict(imports='datasets.consep_dataset')
img_scale = (512, 512)
data_preprocessor = dict(size=img_scale)
optimizer = dict(lr=0.01)
optim_wrapper = dict(optimizer=optimizer)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),
    auxiliary_head=None,
    test_cfg=dict(mode='whole', _delete_=True))
vis_backends = None
visualizer = dict(vis_backends=vis_backends)

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
data_preprocessor = __iv_merge(__base_data_preprocessor, data_preprocessor)
optimizer = __iv_merge(__base_optimizer, optimizer)
optim_wrapper = __iv_merge(__base_optim_wrapper, optim_wrapper)
model = __iv_merge(__base_model, model)
visualizer = __iv_merge(__base_visualizer, visualizer)
if isinstance(img_scale, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    img_scale.pop('_delete_', None)
if isinstance(vis_backends, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    vis_backends.pop('_delete_', None)
