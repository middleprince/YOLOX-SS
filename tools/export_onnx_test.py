#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("-d", "--device", default='cpu', type=str, help="device ")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    optimizer = exp.get_optimizer(1)
    if args.ckpt is None:
        logger.info("===debug info in saveing ckpt===")
        ckpt_state = {
            "start_epoch": 0,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        file_name = os.path.join(exp.output_dir, args.experiment_name)

        if not os.path.exists(file_name):
            os.makedirs(file_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        torch.save(ckpt_state, ckpt_file)
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    if args.device == 'cpu':
        device = torch.device('cpu')
        ckpt = torch.load(ckpt_file, map_location="cpu")
    else: 
        device = torch.device('cuda')
        ckpt = torch.load(ckpt_file, map_location="cuda:0")
        model.half()

    
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.to(device)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    #dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    dummy_input = torch.randn(1, 3, 384, 224)
    if args.device == 'gpu':
        dummy_input = dummy_input.cuda().half()

    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
