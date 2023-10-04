import torch
import torch.nn as nn
from util import AdaBin_Conv2d, Maxout

class BinaryBasicBlock(nn.Module):
    """
    Define the binary basic block, which contains two binary convolution layers.
    """
    expansion = 1  # Expansion factor for the final output

    def __init__(self, in_channels, out_channels, stride=1):
        super(BinaryBasicBlock, self).__init__()
        self.conv1 = AdaBin_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = Maxout(out_channels)

        self.conv2 = AdaBin_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = Maxout(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = self.nonlinear1(out)
        residual = out
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.nonlinear2(out)
        return out


class BinaryResNet(nn.Module):
    """
    Define the binary ResNet architecture.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(BinaryResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = Maxout(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.fc(out)
        return out  # Softmax can be applied later or during the loss computation


def binary_resnet18():
    """
    Helper function to create and return a binary ResNet-18 network.
    """
    return BinaryResNet(BinaryBasicBlock, [2, 2, 2, 2])
