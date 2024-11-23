#!/usr/bin/env python3

import torch.nn as nn

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor ** 2),
                     kernel_size,
                     padding=1,
                     bias=bias)  
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])