#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

from loguru import logger
from .shuffle_net_simple import ShuffleNetV2
from .network_blocks import BaseConv, DWConv, Bottleneck
from ..utils import get_model_info
from torchstat import stat


class YOLOSPAFPN(nn.Module):
    """
    YOLOX model. ShuffleNetV2  is the backbone this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("shuffle3", "shuffle4", "shuffle5"),
        in_channels=[128, 256, 512],
        depthwise=True,
        act='relu' 
    ):
        super().__init__()
        # TODO: using refactorized shufflenetv2 
        self.backbone = ShuffleNetV2(width)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )

        self.C3_p4 = Bottleneck(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            depthwise=depthwise,
            act=act
        )
        #CSPLayer(
        #    int(2 * in_channels[1] * width),
        #    int(in_channels[1] * width),
        #    round(3 * depth),
        #    False,
        #    depthwise=depthwise,
        #    act=act,
        #)  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = Bottleneck(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            depthwise=depthwise,
            act=act
        )
        #CSPLayer(
        #    int(2 * in_channels[0] * width),
        #    int(in_channels[0] * width),
        #    round(3 * depth),
        #    False,
        #    depthwise=depthwise,
        #    act=act,
        #)

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = Bottleneck(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            depthwise=depthwise,
            act=act
        ) 
        #CSPLayer(
        #    int(2 * in_channels[0] * width),
        #    int(in_channels[1] * width),
        #    round(3 * depth),
        #    False,
        #    depthwise=depthwise,
        #    act=act,
        #)

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = Bottleneck(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            depthwise=depthwise,
            act=act
        )

        #CSPLayer(
        #    int(2 * in_channels[1] * width),
        #    int(in_channels[2] * width),
        #    round(3 * depth),
        #    False,
        #    depthwise=depthwise,
        #    act=act,
        #)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # pafpn, which conventional inchannels as [256, 512, 1024]
        # for shufflenetV2, in channels is [128, 256, 512]
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

if __name__ == "__main__":
    model = YOLOSPAFPN(width=0.375)
    img_size = (224, 224)
    logger.info("YOLOSPAFPN Summary:{}".format(stat(model, (3, 224, 224))))
    logger.info("YOLOSPAFPN Summary:{}".format(get_model_info(model, img_size)))
   # print(model)
   # test_data = torch.rand(2, 3, 320, 320)
   # test_outputs = model(test_data)
   # for v in test_outputs:
   #     print(np.ndim(test_outputs.detach().numpy()))
    

