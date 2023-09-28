import  torch
import  torch.nn as nn
import  torch.nn.functional as F
# OPS is a set of layers with same input/output channel.


OPS = {
    'none':         lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity(),
    'avg_pool_3':   lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5':   lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2, count_include_pad=False),
    'avg_pool_7':   lambda C, stride, affine: nn.AvgPool1d(7, stride=stride, padding=3, count_include_pad=False),
    'max_pool_3':   lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'max_pool_5':   lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2),
    'max_pool_7':   lambda C, stride, affine: nn.MaxPool1d(7, stride=stride, padding=3),
    'conv_3':   lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 3, stride, 1, affine=affine),
    'conv_5':   lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 2, affine=affine),
    'conv_7':   lambda C, stride, affine: ReLUConvBN(C, C, 7, stride, 3, affine=affine),
    'sep_conv_3':   lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5':   lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7':   lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3':   lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5':   lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine)
}

class ConvBN(nn.Module):
    """
    Stack of conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        
        super(ConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class ReLUConvBN(nn.Module):
    """
    Stack of relu-conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """
    relu-dilated conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """
    implemented separate convolution via pytorch groups parameters
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """
    zero by stride
    """
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)

