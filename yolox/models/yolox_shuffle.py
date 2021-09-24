#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch
from loguru import logger

from .yolo_shuffle_head import YOLOXSHead
from .yolo_shuffle_pafpn import YOLOSPAFPN
from ..utils import get_model_info
from torchstat import stat


class YOLOXS(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOSPAFPN(width=0.375)
        if head is None:
            head = YOLOXSHead(num_classes=1, width=0.375, depthwise=True)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

if __name__ == "__main__":
    model = YOLOXS()
    img_size = (384, 224)
    img_size2 = (3, 384, 224)
    logger.info("Model Summary")
    stat(model, img_size2)
    logger.info("detector Model Summary:{}".format(get_model_info(model, img_size)))
    #model.cuda()
    #model.eval()

   # test_data = torch.rand(1, 3, 320, 320)
   # with torch.no_grad():
   #     test_outputs = model(test_data) 
   #     print(f"the size of head is:{test_outputs.size()}")
