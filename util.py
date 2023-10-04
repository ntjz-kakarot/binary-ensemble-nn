import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def stochastic_quantization(tensor):
    """
    Stochastic quantization: Maps tensor values to either -1 or 1 based on probabilistic rules.
    """
    return (tensor + 1.).div(2.).add(torch.rand(tensor.size()).to(tensor.device).add(-0.5)).clamp(0., 1.).round().mul(2.).add(-1.)

class BinaryOperation:
    """
    Handles binarization operations for the parameters of a neural network.
    """
    def __init__(self, model, mode='allbin'):
        """
        Constructor: Initializes the object, counts Conv2d layers, and prepares for binarization based on mode.
        """
        count_conv2d = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        assert mode in ['allbin', 'nin'], 'No such mode!'
        start_range, end_range = (0, count_conv2d - 1) if mode == 'allbin' else (1, count_conv2d - 2)

        self.bin_range = np.linspace(start_range, end_range, end_range - start_range + 1).astype('int').tolist()
        self.saved_params = [m.weight.data.clone() for m in model.modules() if isinstance(m, nn.Conv2d) and m in self.bin_range]
        self.target_modules = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d) and m in self.bin_range]

    def binarization(self, quant_mode='det'):
        """
        Binarize the convolution parameters using either deterministic or stochastic quantization.
        """
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams(quant_mode)

    def meancenterConvParams(self):
        """
        Centers the parameters of convolutional layers around zero.
        """
        for module in self.target_modules:
            negMean = module.data.mean(1, keepdim=True).mul(-1).expand_as(module.data)
            module.data.add_(negMean)

    def clampConvParams(self):
        """
        Clamps the convolution parameters between -1 and 1.
        """
        for module in self.target_modules:
            module.data.clamp_(min=-1.0, max=1.0)

    def save_params(self):
        """
        Saves the current full-precision parameters.
        """
        for idx, module in enumerate(self.target_modules):
            self.saved_params[idx].copy_(module.data)

    def binarizeConvParams(self, quant_mode):
        """
        Binarizes the parameters of convolutional layers.
        """
        assert quant_mode in ['det', 'sto'], 'Invalid quantization mode'

        for module in self.target_modules:
            n = module.data[0].nelement()
            s = module.data.size()
            m = module.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            if quant_mode == 'det':
                module.data.mul_(m.expand(s)).sign_()
            elif quant_mode == 'sto':
                module.data = sto_quant(module.data).mul(m.expand(s))

    def restore(self):
        """
        Restores the full-precision values back to the weights.
        """
        for idx, module in enumerate(self.target_modules):
            module.data.copy_(self.saved_params[idx])

    def updateBinaryGradWeight(self):
        """
        Updates the gradients for the binarized weights.
        """
        for module in self.target_modules:
            weight = module.data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m.mul_(module.grad.data)
            m_add = weight.sign().mul(module.grad.data).sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s).mul(weight.sign())
            module.grad.data.add_(m_add).mul_(1.0 - 1.0/s[1]).mul_(n)

class CustomWeightedLoss(nn.Module):
    """
    Custom loss function that is weighted.
    """
    def __init__(self, aggregate='mean'):
        """
        Constructor: Initializes the type of aggregation.
        """
        super(CustomWeightedLoss, self).__init__()
        assert aggregate in ['normal_ce_mean', 's_ce_mean', 'sc_ce_mean'], 'Invalid mode'
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        """
        Computes the loss based on input, target, and optional weights.
        """
        if self.aggregate == 'normal_ce_mean':
            return F.cross_entropy(input, target)
        elif self.aggregate == 's_ce_mean':
            sep_loss = F.cross_entropy(input, target, reduction='none')
            return (sep_loss * weights.to(input.device)).mean()
        elif self.aggregate == 'sc_ce_mean':
            batch_size = target.size(0)
            oned_weights = weights.gather(1, target.unsqueeze(1)).squeeze()
            sep_loss = F.cross_entropy(input, target, reduction='none')
            return (sep_loss * oned_weights.to(input.device)).mean()
