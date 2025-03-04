# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
