'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch
import torch.nn as nn

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__)) # A2J
MODEL_PATH = os.path.join(DIR_PATH, os.path.pardir) # Model
ROOT_PATH = os.path.join(MODEL_PATH, os.path.pardir) # root

sys.path.append(ROOT_PATH)

# Import Project Library
from model.A2J.back_bone.resnet import ResnetBackbone
from model.A2J.back_bone.mobilenet import MobileNet
from model.A2J.a2j_utilities.a2j_branchs import DepthRegression, OffsetRegression, JointClassification

A2J_BACKBONE_CONFIG = {
    "resnet18": {"backbone": ResnetBackbone, "common_trunk": 256, "Regression_trunk": 512},
    "resnet34": {"backbone": ResnetBackbone, "common_trunk": 256, "Regression_trunk": 512},
    "resnet50": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "resnet101": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "resnet152": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "mobilenet": {"backbone": MobileNet, "common_trunk": 512, "Regression_trunk": 1024},
}

class BacknoneNetwork(nn.Module):
    """
    Backbone Network Base Class"
    """
    def __init__(self, backbone_name="resnet18", backbone_pretrained=True):
        """
        Class constructor

        :param backbone_name: the name of the backbone network
        :param backbone_pretrained: load a pretrained backbone network
        """
        super(BacknoneNetwork, self).__init__()
        self.model = A2J_BACKBONE_CONFIG[backbone_name]["backbone"](backbone_name="resnet18", backbone_pretrained=True)
    
    def forward(self, x):
        x1, x2 = self.model(x)
        return x1, x2

class A2J(nn.Module):
    """
    A2J model class
    """
    def __init__(self, num_joints=18, backbone_name="resnet18", backbone_pretrained=True):
        """
        Class constructor

        :param num_joints: number of joints to predict
        :param backbone_name: the name of the backbone network
        :param backbone_pretrained: load a pretrained backbone network
        """
        super(A2J, self).__init__()
        Backbone_Model = A2J_BACKBONE_CONFIG[backbone_name]["backbone"]
        
        self.back_bone = Backbone_Model(name=backbone_name, pretrained=backbone_pretrained)

        self.offset_regression = OffsetRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        self.depth_regression = DepthRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        self.joint_classification = JointClassification(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["common_trunk"], num_joints=num_joints)
        
        # self.Backbone = Backbone_Model(name=backbone_name, pretrained=backbone_pretrained)

        # self.regressionModel = OffsetRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        # self.DepthRegressionModel = DepthRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        # self.classificationModel = JointClassification(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["common_trunk"], num_joints=num_joints)
    
    
    def forward(self, x):
        out3, out4 = self.back_bone(x)
        offset_regression = self.offset_regression(out4)
        depth_regression = self.depth_regression(out4)
        joint_classification = self.joint_classification(out3)

        # out3, out4 = self.Backbone(x)
        # offset_regression = self.regressionModel(out4)
        # depth_regression = self.DepthRegressionModel(out4)
        # joint_classification = self.classificationModel(out3)


        return joint_classification, offset_regression, depth_regression
