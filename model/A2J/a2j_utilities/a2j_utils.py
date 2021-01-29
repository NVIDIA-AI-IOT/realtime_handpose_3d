'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__)) # a2j_utilities
A2J_PATH = os.path.join(DIR_PATH, os.path.pardir) # A2J
MODEL_PATH = os.path.join(A2J_PATH, os.path.pardir) # model
ROOT_PATH = os.path.join(MODEL_PATH, os.path.pardir) # root
sys.path.append(ROOT_PATH)

# Import Project Libraries
import pipeline.constants as const



def generate_anchors(p_h=None, p_w=None):
    """
    Generate anchor shape

    :param p_h: anchor hieght layout
    :param p_w: anchor width layout
    """
    if p_h is None:
        p_h = np.array([2, 6, 10, 14])
    
    if p_w is None:
        p_w = np.array([2, 6, 10, 14])
    
    num_anchors = len(p_h) * len(p_w)

    # Initialize the anchor points
    k = 0
    anchors = np.zeros((num_anchors, 2))
    for i in range(len(p_w)):
        for j in range(len(p_h)):
            anchors[k,1] = p_w[j]
            anchors[k,0] = p_h[i]
            k += 1
    return anchors

def shift(shape, stride, anchor):
    """
    Create the locations of all the anchonrs in the in put image

    :param shape: common trunk (H, W)
    :param stride: the downsampling factor from input to common trunk
    :param anchor: anchor 
    """
    shift_h = np.arange(0, shape[0]) * stride # (shape[0]) 10
    shift_w = np.arange(0, shape[1]) * stride # (shape[1]) 9

    shift_h, shift_w = np.meshgrid(shift_h, shift_w) # (shape[1], shape[0]) (9, 10), (shape[1], shape[0]) (9, 10)
    shifts = np.vstack( (shift_h.ravel(), shift_w.ravel()) ).transpose() # (shape[0]*shape[1], 2) (90, 2)

    A = anchor.shape[0] # 16
    K = shifts.shape[0] # (shape[0]*shape[1]) (90)

    all_anchors = (anchor.reshape(1,A,2) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))) # (shape[0]*shape[1], A, 2)
    all_anchors = all_anchors.reshape((K*A, 2)) # (shape[0]*shape[1]*A, 2)
    return all_anchors
