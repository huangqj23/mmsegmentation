# Converted from configs/segmenter/segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .segmenter_vit_s_mask_8xb1_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmseg.models import CrossEntropyLoss, FCNHead

model.merge(dict(
    decode_head=dict(
        _delete_=True,
        type=FCNHead,
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=150,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0))))
