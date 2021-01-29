'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch.nn as nn

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__)) # a2j_utilities
A2J_PATH = os.path.join(DIR_PATH, os.path.pardir) # A2J
MODEL_PATH = os.path.join(A2J_PATH, os.path.pardir) # model
ROOT_PATH = os.path.join(MODEL_PATH, os.path.pardir) # root
sys.path.append(ROOT_PATH)

# Import Project Library
from model.A2J.back_bone.resnet import get_ResNet

class DepthRegression(nn.Module):
    """
    Depth regression module

    regress the depth of the joints from the anchor points
    """
    def __init__(self, input_channels, output_channels=256, num_anchors=16, num_joints=18):
        """
        Class initializer

        :param input_channels: number of input channels
        :param output_channels: number of output channels
        :param num_anchors: total number of anchor points
        :param num_joints: total number of joints to predict
        """
        super(DepthRegression, self).__init__()
        self.num_joints = num_joints
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels)

        self.output = nn.Conv2d(output_channels, num_anchors*num_joints, kernel_size=3, padding=1)

        # Activation Function
        self.relu = nn.LeakyReLU(inplace=True)

        self._initialize()
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
         # (N, inChannels, 10, 9)
        out = self.conv1(x) # (N, 256, 10, 9)
        out = self.bn1(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv2(out) # (N, 256, 10, 9)
        out = self.bn2(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv3(out) # (N, 256, 10, 9)
        out = self.bn3(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv4(out) # (N, 256, 10, 9)
        out = self.bn4(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.output(out) # (N, num_joints*num_anchors, 10, 9)

        out = out.permute(0, 3, 2, 1) # (N, 9, 10, num_joints*num_anchors)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_joints) # (N, 9, 10, num_anchors, num_joints)
        return out.contiguous().view(batch_size, -1, self.num_joints) # (N, 9*10*num_anchors, num_joint)

class OffsetRegression(nn.Module):
    """
    Offset Regression class

    estimate the joint offsets from the anchorpoints
    """
    def __init__(self, input_channels, output_channels=256, num_anchors=16, num_joints=18):
        """
        Class initializer

        :param input_channels: number of input channels
        :param output_channels: number of output channels
        :param num_anchors: total number of anchor points
        :param num_joints: total number of joints to predict
        """
        super(OffsetRegression, self).__init__()
        
        self.num_anchors = num_anchors
        self.num_joints = num_joints

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels)

        self.output = nn.Conv2d(output_channels, num_anchors*num_joints*2, kernel_size=3, padding=1)

        # Activation Function
        self.relu = nn.LeakyReLU(inplace=True)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def forward(self, x):
        out = self.conv1(x) # (N, 256, 10, 9)
        out = self.bn1(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv2(out) # (N, 256, 10, 9)
        out = self.bn2(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv3(out) # (N, 256, 10, 9)
        out = self.bn3(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.conv4(out) # (N, 256, 10, 9)
        out = self.bn4(out) # (N, 256, 10, 9)
        out = self.relu(out) # (N, 256, 10, 9)

        out = self.output(out) # (N, num_joints*num_anchors*2, 10, 9)

        out = out.permute(0, 3, 2, 1) # (N, 9, 10, num_joints*num_anchors*2)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_joints, 2) # (N, 9, 10, num_anchors, num_joints, 2)
        return out.contiguous().view(batch_size, -1, self.num_joints, 2) # (N, 9*10*num_anchors, num_joints, 2)
    
class JointClassification(nn.Module):
    """
    Joint classification class
    """
    def __init__(self, input_channels, output_channels=256, num_anchors=16, num_joints=18):
        """
        Class initializer

        :param input_channels: number of input channels
        :param output_channels: number of output channels
        :param num_anchors: total number of anchor points
        :param num_joints: total number of joints to predict
        """
        super(JointClassification, self).__init__()

        self.num_anchors = num_anchors
        self.num_joints = num_joints
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels)

        self.output = nn.Conv2d(output_channels, num_anchors*num_joints, kernel_size=3, padding=1)

        # Activation Function
        self.relu = nn.LeakyReLU(inplace=True)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.output(out)

        out = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_joints)
        return out.contiguous().view(batch_size, -1, self.num_joints)
