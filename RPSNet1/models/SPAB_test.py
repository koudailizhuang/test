# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode
import torch.nn.functional as F
import torchvision.transforms as transforms

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat


    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

class SPAB(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=None,
                 out_channels=None,
                 bias=False,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * 255

        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out

def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

input_image_path = 'E:\SiAl\Camera MV-SUA33GM#0001-0003-Snapshot-20231116214835-5171190906301_2.JPG'
input_image = Image.open(input_image_path)

preprocess = transforms.Compose([transforms.Resize((200,200)),transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),])

input_tensor = preprocess(input_image).unsqueeze(0)

SPAB_module = SPAB()
with torch.no_grad():
    output_tensor = SPAB_module(input_tensor)

output_image = output_tensor.squeeze(0).permute(1,2,0).numpy()
output_image = (output_image * np.arry([0.229,0.224,0.225]))+np.array([0.485,0.456,0.406])
output_tensor = np.clip(output_image,0,1)

output_image = (output_image * 255).astype(np.uint8)
output_image = Image.fromarray(output_image)
output_image.show()