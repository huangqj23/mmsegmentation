# Converted from configs/unet/unet-s5-d16_pspnet_4xb4-ce-1.0-dice-3.0-40k_hrf-256x256.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .unet_s5_d16_pspnet_4xb4_40k_hrf_256x256 import *  # noqa: F401,F403
model.merge(dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ])))
