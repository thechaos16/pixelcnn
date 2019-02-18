
import torch
import torch.nn.functional as F
from torch import nn
from layers import MaskedConv2d, GatedUnit


class PixelCNN(nn.Module):
    def __init__(self, layer_size, layer_num, gated=False):
        super(PixelCNN, self).__init__()
        self.layer_size = layer_size
        self.layer_num = layer_num
        self._init_input_output_layer()
        self._init_layer()
        self.gated = gated

    def _init_input_output_layer(self):
        self.conv1 = MaskedConv2d('A', False, 1, self.layer_size, 7, 1, 3, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.layer_size)
        self.last_conv = nn.Conv2d(self.layer_size, 256, 1)
        self.last_relu = nn.ReLU(True)

    def _init_layer(self):
        self.b_type_masks = nn.Sequential()
        for i in range(self.layer_num):
            self.b_type_masks.add_module(
                'causal_{}'.format(i), MaskedConv2d('B', False, self.layer_size, self.layer_size, 7, 1, 3, bias=False)
            )
            self.b_type_masks.add_module('bn_{}'.format(i), nn.BatchNorm2d(self.layer_size))
            self.b_type_masks.add_module('relu_{}'.format(i), nn.ReLU(True))

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        if self.gated:
            x = torch.cat((x, x), 1)
        x = self.b_type_masks(x)
        x = self.last_conv(x)
        x = self.last_relu(x)
        return F.log_softmax(x)


class ConditionalSequential(nn.Sequential):
    def __init__(self):
        super(ConditionalSequential, self).__init__()

    def forward(self, input, h):
        for module in self._modules.values():
            input = module(input, h)
        return input


class GatedPixelCNN(PixelCNN):
    def __init__(self, layer_size, layer_num, conditional=False, num_classes=0):
        self.conditional = conditional
        self.num_classes = num_classes
        super(GatedPixelCNN, self).__init__(layer_size, layer_num, gated=True)

    def _init_input_output_layer(self):
        self.conv1 = MaskedConv2d('A', False, 1, self.layer_size, 7, 1, 3)
        # self.conv1 = GatedUnitSep(self.layer_size, kernel_size=7, first_layer=True)
        self.conv1_bn = nn.BatchNorm2d(self.layer_size)
        self.last_conv = nn.Conv2d(self.layer_size * 2, 256, 1)
        self.last_relu = nn.ReLU(True)

    def _init_layer(self):
        if self.conditional:
            self.b_type_masks = ConditionalSequential()
        else:
            self.b_type_masks = nn.Sequential()
        for i in range(self.layer_num):
            self.b_type_masks.add_module(
                'gated_unit_{}'.format(i),
                GatedUnit(self.layer_size, conditional=self.conditional, num_classes=self.num_classes)
            )

    def forward(self, x, h=None):
        if self.conditional:
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = torch.cat((x, x), 1)
            x = self.b_type_masks(x, h)
            x = self.last_conv(x)
            x = self.last_relu(x)
            return F.log_softmax(x)
        return super(GatedPixelCNN, self).forward(x)
