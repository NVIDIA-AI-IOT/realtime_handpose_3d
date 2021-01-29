'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

PRETRAINED_MODELS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    """
    Resnet Basic Residual Block
    """
    expansion = 1
    def __init__(self, input_channels, output_channels, stride=1, dilation=1, downsample=None):
        """
        Class constructor

        :param input_channels: number of input channels to the residual block
        :param output channels: number of putput channels of the residual block
        :param stride: stride of the first convolution in the residual block
        :param dilation: dilation of the second convolution in the residual block
        :param downsample: torch.nn function for down sampling the input x for concatenation in the residual layer
        """
        super(BasicBlock, self).__init__()
    
        self.conv1 = conv3x3(input_channels, output_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = conv3x3(output_channels, output_channels, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.downsample = downsample
        self.stride = stride

        # Actiation function
        self.relu = nn.LeakyReLU(inplace=True)

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    Resnet Bottleneck network
    """
    expansion = 4
    def __init__(self, input_channels, output_channels, stride=1, dilation=1, downsample=None):
        """
        Class constructor

        :param input_channels: number of input channels to the residual block
        :param output channels: number of putput channels of the residual block
        :param stride: stride of the second convolution in the residual block
        :param dilation: dilation of the second convolution in the residual block
        :param downsample: torch.nn function for down sampling the input x for concatenation in the residual layer
        """
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(input_channels, output_channels)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = conv3x3(output_channels, output_channels, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.conv3 = conv1x1(output_channels, output_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(output_channels*self.expansion)

        self.downsample = downsample
        self.stride = stride

        # Activation function
        self.relu = nn.LeakyReLU(inplace=True)


    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet Definition

    could create resnet (18, 34, 50, 101, 152) by setting the parameters
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        """
        Class constructor

        :param block: type toch.nn, A residual block class instance (i.e. BasicBlock or Bottleneck)
        :param layers: type list, A list holding the number of residual blocks in each ResNet layer
        :param num_classes: if using a pretrained network make sure the number of classes are the same
        :param zero_init_residual: Zero Initialiaze the last batchnorm in each residual layer for higher accuracy
        """
        super(ResNet, self).__init__()
        
        self.input_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_resnet_layer(block, 64, layers[0])
        self.layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_resnet_layer(block, 512, layers[3], stride=1, dilation=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        # Activation function
        self.relu = nn.LeakyReLU(inplace=True)

        self._initialize()
        if zero_init_residual:
            self._zero_initialize()
            
    def _make_resnet_layer(self, block, output_channels, blocks, stride=1, dilation=1):
        """
        Method to create residual block layer in resnet

        :param block: type torch.nn, a residual block block class instance (i.e. BasicBlock or Bottleneck)
        :param output_channels: type int, number of output channels of the residual block layer
        :param blocks: type int, number of residual blocks in this layer
        :param stride: type int
        :param dilation: type int
        """
        downsample = None
        
        if (stride != 1) or (self.input_channels != output_channels*block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.input_channels, output_channels*block.expansion, stride=stride),
                nn.BatchNorm2d(output_channels*block.expansion),
            )
        
        layers = list()
        layers.append(block(self.input_channels, output_channels, stride=stride, downsample=downsample))

        self.input_channels = output_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.input_channels, output_channels, dilation=dilation))
        
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _zero_initialize(self):
        for m in self.modules():
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_ResNet(resnet_model="resnet18", pretrained=False):
    
    resnet_setups = {
        "resnet18": {"block": BasicBlock, "layers": [2, 2, 2, 2]},
        "resnet34": {"block": BasicBlock, "layers": [3, 4, 6, 3]},
        "resnet50": {"block": Bottleneck, "layers": [3, 4, 6, 3]},
        "resnet101": {"block": Bottleneck, "layers": [3, 4, 23, 3]},
        "resnet152": {"block": Bottleneck, "layers": [3, 8, 36, 3]},
    }
    model = ResNet(resnet_setups[resnet_model]["block"], resnet_setups[resnet_model]["layers"])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(PRETRAINED_MODELS[resnet_model]))
    
    return model

class ResnetBackbone(nn.Module):
    """
    The Resnet Backbone module
    """
    def __init__(self, name="resnet18", pretrained=True):
        """
        Class constructor

        :param name: name of the resnet model to load
        :param pretrained: weather or not to load the weight of a pretrained model on ImageNet
        """
        super(ResnetBackbone, self).__init__()
        self.model = get_ResNet(resnet_model=name, pretrained=pretrained)
    
    def forward(self, x):
        n, c, h, w = x.size()

        x = x[:,0:1,:,:] # depth
        x = x.expand(n, 3, h, w)

        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)

        out1 = self.model.layer1(out)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)

        return out3, out4
