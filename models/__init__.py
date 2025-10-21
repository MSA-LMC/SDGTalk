#!/usr/bin/env python

from .SDGTalk import FlameDriveAvatar, AudioDriveAvatar

def build_model(model_cfg, ):
    model_dict = {
        'FlameDriveAvatar': FlameDriveAvatar,
        'AudioDriveAvatar': AudioDriveAvatar,
    }
    return model_dict[model_cfg.NAME](model_cfg, )
