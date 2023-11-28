import torch.nn as nn
import torch

class BinActive(torch.autograd.Function):
    """
    Binarize the input activations and calculate the mean across the channel dimension.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        mean = torch.mean(input.abs(), 1, keepdim=True)
        bin_input = input.sign()
        return bin_input, mean

    @staticmethod
    def backward(ctx, grad_output, grad_output_mean):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    """
    Convolutional layer with binarized weights and inputs.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dropout=0):
        super(BinConv2d, self).__init__()
        self.dropout_ratio = dropout
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x, _ = BinActive.apply(x)
        
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        
        x = self.conv(x)
        x = self.relu(x)
        return x

class RealConv2d(nn.Module):
    """
    Standard convolutional layer with float values.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dropout=0):
        super(RealConv2d, self).__init__()
        self.dropout_ratio = dropout
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        
        x = self.conv(x)
        x = self.relu(x)
        return x

class BaseNet(nn.Module):
    """
    Base model that can be used for both the binarized and real-valued versions.
    """
    def __init__(self, Conv2dLayer, cut_ratio=1):
        super(BaseNet, self).__init__()
        self.xnor = nn.Sequential(
            Conv2dLayer(3, int(192*cut_ratio), 5, 1, 2),
            Conv2dLayer(int(192*cut_ratio), int(96*cut_ratio), 1, 1, 0),
            nn.MaxPool2d(3, 2, 1),
            Conv2dLayer(int(96*cut_ratio), int(192*cut_ratio), 5, 1, 2, 0.5),
            Conv2dLayer(int(192*cut_ratio), int(192*cut_ratio), 1, 1, 0),
            nn.AvgPool2d(3, 2, 1),
            Conv2dLayer(int(192*cut_ratio), int(192*cut_ratio), 3, 1, 1, 0.5),
            Conv2dLayer(int(192*cut_ratio), 10, 1, 1, 0),
            nn.AvgPool2d(8, 1, 0)
        )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x

# Define specific model variants
class Net(BaseNet):
    def __init__(self):
        super(Net, self).__init__(BinConv2d)

class RealNet(BaseNet):
    def __init__(self):
        super(RealNet, self).__init__(RealConv2d)

class Net_Cut(BaseNet):
    def __init__(self, cut_ratio=0.5):
        super(Net_Cut, self).__init__(BinConv2d, cut_ratio)

class RealNet_Cut(BaseNet):
    def __init__(self, cut_ratio=0.5):
        super(RealNet_Cut, self).__init__(RealConv2d, cut_ratio)
