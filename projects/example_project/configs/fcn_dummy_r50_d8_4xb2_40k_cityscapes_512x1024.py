# Converted from projects/example_project/configs/fcn_dummy-r50-d8_4xb2-40k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from mmseg.configs.fcn.fcn_r50_d8_4xb2_40k_cityscapes_512x1024 import *  # noqa: F401,F403

from projects.example_project.dummy import DummyResNet

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

crop_size = __iv_scope(crop_size, 'mmseg')
data_preprocessor = __iv_scope(data_preprocessor, 'mmseg')
data_root = __iv_scope(data_root, 'mmseg')
dataset_type = __iv_scope(dataset_type, 'mmseg')
default_hooks = __iv_scope(default_hooks, 'mmseg')
default_scope = __iv_scope(default_scope, 'mmseg')
env_cfg = __iv_scope(env_cfg, 'mmseg')
img_ratios = __iv_scope(img_ratios, 'mmseg')
load_from = __iv_scope(load_from, 'mmseg')
log_level = __iv_scope(log_level, 'mmseg')
log_processor = __iv_scope(log_processor, 'mmseg')
model = __iv_scope(model, 'mmseg')
norm_cfg = __iv_scope(norm_cfg, 'mmseg')
optim_wrapper = __iv_scope(optim_wrapper, 'mmseg')
optimizer = __iv_scope(optimizer, 'mmseg')
param_scheduler = __iv_scope(param_scheduler, 'mmseg')
resume = __iv_scope(resume, 'mmseg')
test_cfg = __iv_scope(test_cfg, 'mmseg')
test_dataloader = __iv_scope(test_dataloader, 'mmseg')
test_evaluator = __iv_scope(test_evaluator, 'mmseg')
test_pipeline = __iv_scope(test_pipeline, 'mmseg')
train_cfg = __iv_scope(train_cfg, 'mmseg')
train_dataloader = __iv_scope(train_dataloader, 'mmseg')
train_pipeline = __iv_scope(train_pipeline, 'mmseg')
tta_model = __iv_scope(tta_model, 'mmseg')
tta_pipeline = __iv_scope(tta_pipeline, 'mmseg')
val_cfg = __iv_scope(val_cfg, 'mmseg')
val_dataloader = __iv_scope(val_dataloader, 'mmseg')
val_evaluator = __iv_scope(val_evaluator, 'mmseg')
vis_backends = __iv_scope(vis_backends, 'mmseg')
visualizer = __iv_scope(visualizer, 'mmseg')


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

custom_imports = dict(imports=['dummy'])

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, backbone=dict(type=DummyResNet))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
data_preprocessor = __iv_merge(__base_data_preprocessor, data_preprocessor)
model = __iv_merge(__base_model, model)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
