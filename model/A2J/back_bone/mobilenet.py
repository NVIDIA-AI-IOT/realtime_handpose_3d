'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from math import sqrt
from itertools import product as product

# Set the global device variable to cuda is GPU is avalible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNet(nn.Module):
    """
    MobileNet Bass class to produce lower lever features
    """
    def __init__(self, **kwargs):
        super(MobileNet, self).__init__()

        # Activation function
        self.relu = nn.LeakyReLU(0.01)

        # Standard MobileNet Convolution layers
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.bn1_4 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.bn2_4 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.bn3_3 = nn.BatchNorm2d(512)
        self.conv3_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn3_4 = nn.BatchNorm2d(512)
        self.conv3_5 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn3_5 = nn.BatchNorm2d(512)
        self.conv3_6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn3_6 = nn.BatchNorm2d(512)
        self.conv3_7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn3_7 = nn.BatchNorm2d(512) #    <---
        self.conv3_8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512)
        self.bn3_8 = nn.BatchNorm2d(512)
        
        self.conv4_1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_1 = nn.BatchNorm2d(1024)
        self.conv4_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn4_2 = nn.BatchNorm2d(1024)
        self.conv4_3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_3 = nn.BatchNorm2d(1024)
        self.conv4_4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn4_4 = nn.BatchNorm2d(1024)
        self.conv4_5 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_5 = nn.BatchNorm2d(1024)
        self.conv4_6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn4_6 = nn.BatchNorm2d(1024)
        self.conv4_7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_7 = nn.BatchNorm2d(1024)
        self.conv4_8 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn4_8 = nn.BatchNorm2d(1024)
        self.conv4_9 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_9 = nn.BatchNorm2d(1024)
        self.conv4_10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn4_10 = nn.BatchNorm2d(1024)
        self.conv4_11 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn4_11 = nn.BatchNorm2d(1024) #    <---

        self._init_conv2d()
    
    def _init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


    def forward(self, x):
        out = self.relu(self.conv1_1(x)) # (N, 32, 150, 150)
        out = self.bn1_1(out) # (N, 32, 150, 150)
        out = self.relu(self.conv1_2(out)) # (N, 32, 150, 150)
        out = self.bn1_2(out) # (N, 32, 150, 150)
        out = self.relu(self.conv1_3(out)) # (N, 64, 150, 150)
        out = self.bn1_3(out) # (N, 64, 150, 150)
        
        out = self.relu(self.conv1_4(out)) # (N, 64, 75, 75)
        out = self.bn1_4(out) # (N, 64, 75, 75)
        out = self.relu(self.conv2_1(out)) # (N, 128, 75, 75)
        out = self.bn2_1(out) # (N, 128, 75, 75)
        out = self.relu(self.conv2_2(out)) # (N, 128, 75, 75)
        out = self.bn2_2(out) # (N, 128, 75, 75)
        out = self.relu(self.conv2_3(out)) # (N, 128, 75, 75)
        out = self.bn2_3(out) # (N, 128, 75, 75)
        
        out = self.relu(self.conv2_4(out)) # (N, 128, 38, 38)
        out = self.bn2_4(out) # (N, 128, 38, 38)
        out = self.relu(self.conv3_1(out)) # (N, 256, 38, 38)
        out = self.bn3_1(out) # (N, 256, 38, 38)
        out = self.relu(self.conv3_2(out)) # (N, 256, 38, 38)
        out = self.bn3_2(out) # (N, 256, 38, 38)
        out = self.relu(self.conv3_3(out)) # (N, 256, 38, 38)
        out = self.bn3_3(out) # (N, 512, 38, 38)
        out = self.relu(self.conv3_4(out)) # (N, 512, 38, 38)
        out = self.bn3_4(out) # (N, 512, 38, 38)
        out = self.relu(self.conv3_5(out)) # (N, 512, 38, 38)
        out = self.bn3_5(out) # (N, 512, 38, 38)
        out = self.relu(self.conv3_6(out)) # (N, 512, 38, 38)
        out = self.bn3_6(out) # (N, 512, 38, 38)
        out = self.relu(self.conv3_7(out)) # (N, 512, 38, 38)
        out = self.bn3_7(out) # (N, 512, 38, 38)
        out = self.relu(self.conv3_8(out)) # (N, 512, 19, 19)
        out = self.bn3_8(out) # (N, 256, 19, 19)
        conv3_8 = out 
        
        out = self.relu(self.conv4_1(out)) # (N, 1024, 19, 19)
        out = self.bn4_1(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_2(out)) # (N, 1024, 19, 19)
        out = self.bn4_2(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_3(out)) # (N, 1024, 19, 19)
        out = self.bn4_3(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_4(out)) # (N, 1024, 19, 19)
        out = self.bn4_4(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_5(out)) # (N, 1024, 19, 19)
        out = self.bn4_5(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_6(out)) # (N, 1024, 19, 19)
        out = self.bn4_6(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_7(out)) # (N, 1024, 19, 19)
        out = self.bn4_7(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_8(out)) # (N, 1024, 19, 19)
        out = self.bn4_8(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_9(out)) # (N, 1024, 19, 19)
        out = self.bn4_9(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_10(out)) # (N, 1024, 19, 19)
        out = self.bn4_10(out) # (N, 1024, 19, 19)
        out = self.relu(self.conv4_11(out)) # (N, 1024, 19, 19)
        out = self.bn4_11(out) # (N, 1024, 19, 19)    <-----
        conv12_4 = out

        return conv3_8, conv12_4
