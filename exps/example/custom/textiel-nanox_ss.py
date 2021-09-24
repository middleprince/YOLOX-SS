#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os

import torch.nn as nn
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ----------- dataloader config------- #
        self.input_size = (960, 960);
        self.random_size = (10, 20)
        self.data_dir = os.getenv("YOLOX_DIR", None)
        self.train_ann = "train.json" # for training json 
        self.val_ann = "val.json"

        # ------------model config -------------------- #
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.375

        # -----------------transform config---------
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.8, 1.6)
        self.mixup_scale = (0.5, 1.5)
        #self.scale = (0.1, 2)
        self.mosaic_prob = 1;
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 20
        self.min_lr_ratio = 0.1
        self.ema = True
 
        self.weight_decay = 5e-4
        self.momentum = 0.90
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
 
        # -----------------  testing config ------------------ #
        # h * w not w * h
        #self.test_size = (384, 384)
        self.test_size = (640, 640)
        #self.test_size = (416, 416)
        #self.test_size = (224, 416)
        #self.test_size = (224, 384)
        #self.test_size = (384, 384)
        #self.test_size = (288, 512)
        #self.test_size = (288, 512)
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # -----------------  weight save  config ------------------ #
        self.eval_interval = 10
        self.print_interval = 10

    # nano using depthwise conv 
    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOXSSHead, YOLOSPAFPN, YOLOXSS
            in_channels = [128, 256, 512]
            backbone = YOLOSPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXSSHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOXSS(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            name="val" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    
