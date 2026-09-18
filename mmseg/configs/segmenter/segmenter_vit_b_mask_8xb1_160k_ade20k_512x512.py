# Converted from configs/segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.segmenter_vit_b16_mask import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403

from mmengine.optim import OptimWrapper

crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(data_preprocessor=data_preprocessor))
optimizer.merge(dict(lr=0.001, weight_decay=0.0))
optim_wrapper.merge(dict(type=OptimWrapper, optimizer=optimizer))
train_dataloader.merge(dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1))
val_dataloader.merge(dict(batch_size=1))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
