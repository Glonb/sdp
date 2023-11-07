import  torch
import  torch.nn as nn
import  torch.nn.functional as F
# OPS is a set of layers with same input/output channel.


OPS = {
    'skip_connect':    lambda C: Identity(),
    'conv_3_1':        lambda C: ConvReLU(C, C, 3, 1),
    'conv_4_1':        lambda C: ConvReLU(C, C, 4, 1),
    'conv_5_1':        lambda C: ConvReLU(C, C, 5, 1),
    'conv_6_1':        lambda C: ConvReLU(C, C, 6, 1),
    'conv_7_1':        lambda C: ConvReLU(C, C, 7, 1),
    'conv_9_1':        lambda C: ConvReLU(C, C, 9, 1),
    'conv_3_2':        lambda C: ConvReLU(C, C, 3, 2),
    'conv_4_2':        lambda C: ConvReLU(C, C, 4, 2),
    'conv_5_2':        lambda C: ConvReLU(C, C, 5, 2),
    'conv_6_2':        lambda C: ConvReLU(C, C, 6, 2),
    'conv_7_2':        lambda C: ConvReLU(C, C, 7, 2),
    'conv_9_2':        lambda C: ConvReLU(C, C, 9, 2),
    'sep_conv_3_1':    lambda C: SepConv(C, C, 3, 1),
    'sep_conv_5_1':    lambda C: SepConv(C, C, 5, 1),
    'sep_conv_7_1':    lambda C: SepConv(C, C, 7, 1),
    'sep_conv_9_1':    lambda C: SepConv(C, C, 9, 1),
    'sep_conv_3_2':    lambda C: SepConv(C, C, 3, 2),
    'sep_conv_5_2':    lambda C: SepConv(C, C, 5, 2),
    'sep_conv_7_2':    lambda C: SepConv(C, C, 7, 2),
    'sep_conv_9_2':    lambda C: SepConv(C, C, 9, 2),
    'dil_conv_3_1':    lambda C: DilConv(C, C, 3, 1, 2),
    'dil_conv_5_1':    lambda C: DilConv(C, C, 5, 1, 2),
    'dil_conv_7_1':    lambda C: DilConv(C, C, 7, 1, 2),
    'avg_pool_3_1':    lambda C: nn.AvgPool1d(kernel_size=3, stride=2, padding=1, count_include_pad=False),
    'avg_pool_5_1':    lambda C: nn.AvgPool1d(kernel_size=5, stride=2, padding=2, count_include_pad=False),
    'avg_pool_7_1':    lambda C: nn.AvgPool1d(kernel_size=7, stride=2, padding=3, count_include_pad=False),
    'max_pool_3_1':    lambda C: nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    'max_pool_5_1':    lambda C: nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
    'max_pool_7_1':    lambda C: nn.MaxPool1d(kernel_size=7, stride=2, padding=3)
}

class ConvReLU(nn.Module):
    """
    Stack of conv-relu
    """
    def __init__(self, C_in, C_out, kernel_size, stride):
        
        super(ConvReLU, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """
    dilated conv-relu
    """
    def __init__(self, C_in, C_out, kernel_size, stride, dilation):
        
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=C_in),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """
    implemented separate convolution via pytorch groups parameters
    """
    def __init__(self, C_in, C_out, kernel_size, stride):
        
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, groups=C_in),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


# class AvgPool(nn.Module):
    
#     def __init__(self, kernel_size, stride, padding):
        
#         super(AvgPool, self).__init__()

#         self.op = nn.AvgPool1d(kernel_size=kernel_size, stride=2*stride, padding=padding, count_include_pad=False)
           
#     def forward(self, x):
#         x = self.op(x)
#         batch_size, channels, length = x.size()

#         padding_layer = nn.ConstantPad1d(padding=(0, length), value=0.0)
        
#         return padding_layer(x)


# class MaxPoolPadding(nn.Module):
    
#     def __init__(self, kernel_size, stride, padding):
        
#         super(MaxPoolPadding, self).__init__()
        
#         self.op = nn.MaxPool1d(kernel_size=kernel_size, stride=2*stride, padding=padding)

#     def forward(self, x):
#         x = self.op(x)
#         batch_size, channels, length = x.size()
        
#         padding_layer = nn.ConstantPad1d(padding=(0, length), value=0.0)
        
#         return padding_layer(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



