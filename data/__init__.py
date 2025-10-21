#!/usr/bin/env python

from .loader_track_lip import FlameDriveData, AudioDriveData, InferTestData , InferCommonData

def build_dataset(data_cfg, split):
    dataset_dict = {
        'FlameDriveData': FlameDriveData,
        'AudioDriveData': AudioDriveData,
    }
    return dataset_dict[data_cfg.LOADER](data_cfg, split)
