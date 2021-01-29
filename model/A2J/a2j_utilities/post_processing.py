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
import torch.nn as nn
import torch.nn.functional as F

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__)) # a2j_utilities
A2J_PATH = os.path.join(DIR_PATH, os.path.pardir) # A2J
MODEL_PATH = os.path.join(A2J_PATH, os.path.pardir) # model
ROOT_PATH = os.path.join(MODEL_PATH, os.path.pardir) # root
sys.path.append(ROOT_PATH)

# Import Project Library
import pipeline.constants as const
from model.A2J.a2j_utilities.a2j_utils import generate_anchors, shift

class PostProcess(nn.Module):
    """
    PosrProcessing class
    """
    def __init__(self, p_h=None, p_w=None, shape=[const.A2J_TARGET_SIZE[1]//16, const.A2J_TARGET_SIZE[0]//16],\
                    stride=const.A2J_STRIDE):
        """
        Class constructior

        :param p_w: 
        """
        
        super(PostProcess, self).__init__()
        anchors = generate_anchors(p_h=p_h, p_w=p_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)).float()

    def forward(self, joint_classifications, offset_regressions, depth_regressions):
        """
        forward pass through the module

        :param joint_classifications: type torch.tensor, joint classification output of the model
        :param offset_regressions:  type torch.tensor, offset regression output of the model
        :param depth_regressions:  type torch.tensor, depth rgression output of the model
        """
        DEVICE = joint_classifications.device

        batch_size = joint_classifications.shape[0]
        anchor = self.all_anchors.to(DEVICE)  # (shape[0]*shape[1]*anchor_stride, 2) (1440, 2)
        predictions = list()

        for i in range(batch_size):
            joint_classification = joint_classifications[i] # (shape[0]*shape[1]*anchor_stride, num_joints) (1440, 18)
            offset_regression = offset_regressions[i] # (shape[0]*shape[1]*anchor_stride, num_joints, 2) (1440, 18, 2)
            depth_regression = depth_regressions[i] # (shape[0]*shape[1]*anchor_stride, num_joits) (1440, 18)

            # xy_regression: is the location of each anchor point + the offset
            # offset_regression: is giving us the offset
            xy_regression = torch.unsqueeze(anchor, 1).to(DEVICE) + offset_regression # (shape[0]*shape[1]*anchor_stride, 2) (1440, 18, 2)

            # reg_weight: is gining us the classification (importance) of each anchor point
            reg_weight = F.softmax(joint_classification, dim=0) # (shape[0]*shape[1]*anchor_stride, num_joints) (1440, 18)

            # reg_weigh_xy: is reg_weight expanded to have to tensors to multiply to each x and y coordinates
            reg_weight_xy = reg_weight.unsqueeze(2).expand(reg_weight.shape[0], reg_weight.shape[1], 2).to(DEVICE) # (shape[0]*shape[1]*anchor_stride, num_joints, 2) (1440, 18, 2)

            prediction_xy = (reg_weight_xy * xy_regression).sum(0)
            prediction_depth = (reg_weight * depth_regression).sum(0)

            prediction_depth = prediction_depth.unsqueeze(1).to(DEVICE)

            prediction = torch.cat((prediction_xy, prediction_xy), 1)
            predictions.append(prediction)
        
        return predictions
