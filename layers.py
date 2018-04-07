
import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv2d(nn.Conv2d):
    # TODO: expand to multiple channels
    def __init__(self, mask_type, vertical, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kernel_height, kernel_width = self.weight.size()
        center_height = kernel_height // 2
        center_width = kernel_width // 2
        self.mask.fill_(1)
        if not vertical:
            if kernel_height != 1:
                self.mask[:, :, center_height, center_width + (mask_type == 'B'):] = 0
                self.mask[:, :, center_height + 1:, :] = 0
        else:
            # if 1x1 conv, no mask
            if kernel_height != 1:
                # noblind if not the first layer
                # self.mask[:, :, center_height + (mask_type == 'B'):, :] = 0
                self.mask[:, :, center_height:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class GatedConv2d(nn.Module):
    def __init__(self):
        super(GatedConv2d, self).__init__()

    def forward(self, x):
        split_dim = 1
        x_0, x_1 = torch.split(x, x.size()[split_dim]//2, split_dim)
        return F.tanh(x_0) * F.sigmoid(x_1)


class GatedUnit(nn.Module):
    def __init__(self, layer_size, kernel_size=3, first_layer=False, conditional=False, num_classes=0):
        super(GatedUnit, self).__init__()
        self.layer_size = layer_size
        self.conditional = conditional
        padding = kernel_size//2
        mask_type = 'A' if first_layer else 'B'
        # separated nxn vertical unit
        self.vertical_f = MaskedConv2d(mask_type, True, self.layer_size,
                                       self.layer_size, kernel_size, 1, padding)
        self.vertical_g = MaskedConv2d(mask_type, True, self.layer_size,
                                       self.layer_size, kernel_size, 1, padding)
        # conditional matrix
        if self.conditional:
            # NOTE: it does not have to be masked (not sure but written in the paper)
            # linear version
            self.vertical_conditional = nn.Linear(num_classes, self.layer_size)
            self.horizontal_conditional = nn.Linear(num_classes, self.layer_size)
            # TODO: 1x1 convolutional version
            # self.vertical_conditional = nn.Conv2d()
            # self.horizontal_conditional = nn.Conv2d()
        # gated vertical
        self.vertical_gated = GatedConv2d()
        # separated 1x1 vertical unit
        self.vertical_f_11 = MaskedConv2d(mask_type, True, self.layer_size, self.layer_size, 1, 1, 0)
        self.vertical_g_11 = MaskedConv2d(mask_type, True, self.layer_size, self.layer_size, 1, 1, 0)
        # separated nxn horizontal unit
        self.horizontal_f = MaskedConv2d(mask_type, False, self.layer_size,
                                         self.layer_size, kernel_size, 1, padding)
        self.horizontal_g = MaskedConv2d(mask_type, False, self.layer_size,
                                         self.layer_size, kernel_size, 1, padding)
        # gated horizontal
        self.horizontal_gated = GatedConv2d()
        # horizontal 11 conv
        self.horizontal_11 = MaskedConv2d(mask_type, False, self.layer_size, self.layer_size, 1, 1, 0)

    def forward(self, x, h=None):
        split_dim = 1
        x_v, x_h = torch.split(x, x.size()[split_dim]//2, split_dim)
        input_dim = x_v.size()
        x_vert_f = self.vertical_f(x_v)
        x_vert_g = self.vertical_g(x_v)
        if self.conditional:
            vert_cond = self.vertical_conditional(h).unsqueeze(2).unsqueeze(3).expand(input_dim)
            x_vert_f = x_vert_f + vert_cond
            x_vert_g = x_vert_g + vert_cond
        x_vert_gated = self.vertical_gated(torch.cat([x_vert_f, x_vert_g], split_dim))
        x_vert_f_11 = self.vertical_f_11(x_vert_f)
        x_vert_g_11 = self.vertical_g_11(x_vert_g)
        x_hor_f = self.horizontal_f(x_h)
        x_hor_g = self.horizontal_g(x_h)
        if self.conditional:
            hori_cond = self.horizontal_conditional(h).unsqueeze(2).unsqueeze(3).expand(input_dim)
            x_hor_f += hori_cond
            x_hor_g += hori_cond
        x_hor_gated = self.horizontal_gated(torch.cat([x_vert_f_11 + x_hor_f, x_vert_g_11 + x_hor_g], split_dim))
        x_hor_11 = self.horizontal_11(x_hor_gated)
        residual = x_hor_11 + x_h
        return torch.cat([x_vert_gated, residual], split_dim)
