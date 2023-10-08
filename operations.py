import  torch
import  torch.nn as nn
import  torch.nn.functional as F
# OPS is a set of layers with same input/output channel.


OPS = {
    # 'none':         lambda C, stride, affine: Zero(stride),
    'skip_connect':  lambda C, stride: Identity(),
    'conv_3':        lambda C, stride: ConvReLU(C, C, 3, stride, 1),
    'conv_5':        lambda C, stride: ConvReLU(C, C, 5, stride, 2),
    'conv_7':        lambda C, stride: ConvReLU(C, C, 7, stride, 3),
    'sep_conv_3':    lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5':    lambda C, stride: SepConv(C, C, 5, stride, 2),
    'sep_conv_7':    lambda C, stride: SepConv(C, C, 7, stride, 3),
    'dil_conv_3':    lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'dil_conv_5':    lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    'dil_conv_7':    lambda C, stride: DilConv(C, C, 7, stride, 6, 2),
    'avg_pool_3':    lambda C, stride: AvgPoolPadding(3, stride, 1),
    'avg_pool_5':    lambda C, stride: AvgPoolPadding(5, stride, 2),
    'avg_pool_7':    lambda C, stride: AvgPoolPadding(7, stride, 3),
    'max_pool_3':    lambda C, stride: MaxPoolPadding(3, stride, 1),
    'max_pool_5':    lambda C, stride: MaxPoolPadding(5, stride, 2),
    'max_pool_7':    lambda C, stride: MaxPoolPadding(7, stride, 3)
}

class ConvReLU(nn.Module):
    """
    Stack of conv-relu
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        
        super(ConvReLU, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)

# class ReLUConvBN(nn.Module):
#     """
#     Stack of relu-conv-bn
#     """
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        
#         super(ReLUConvBN, self).__init__()

#         self.op = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm1d(C_out, affine=affine)
#         )

#     def forward(self, x):
#         return self.op(x)


class DilConv(nn.Module):
    """
    dilated conv-relu
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=C_in),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """
    implemented separate convolution via pytorch groups parameters
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


class AvgPoolPadding(nn.Module):
    
    def __init__(self, kernel_size, stride, padding):
        
        super(AvgPoolPadding, self).__init__()
        self.padding_size = None

        self.op = nn.Sequential(
            nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False),
            # nn.ConstantPad1d(padding=(0, self.padding_size), value=0.0)
        )

    def forward(self, x):
        batch_size, channels, length = x.size()

        if self.padding_size is None:
            self.padding_size = length
        padding_layer = nn.ConstantPad1d(padding=(0, self.padding_size), value=0.0)
        return padding_layer(self.op(x))


class MaxPoolPadding(nn.Module):
    
    def __init__(self, kernel_size, stride, padding):
        
        super(MaxPoolPadding, self).__init__()

        self.op = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.op(x)
        padding_data = torch.zeros(x.shape[0], x.shape[1], x.shape[2])
        return torch.cat((x, padding_data), dim=2)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
   
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)

