import  torch
import  torch.nn as nn
import torch.nn.init as init
import  torch.nn.functional as F
# OPS is a set of layers with same input/output channel.


OPS = {
    'skip_connect':    lambda C: Identity(),
    'conv_3_1':        lambda C: ConvReLU(C, C // 2, 3, 1),
    'conv_4_1':        lambda C: ConvReLU(C, C // 2, 4, 1),
    'conv_5_1':        lambda C: ConvReLU(C, C // 2, 5, 1),
    'conv_6_1':        lambda C: ConvReLU(C, C // 2, 6, 1),
    'conv_7_1':        lambda C: ConvReLU(C, C // 2, 7, 1),
    'conv_3_2':        lambda C: ConvReLU(C, C // 2, 3, 2),
    'conv_4_2':        lambda C: ConvReLU(C, C // 2, 4, 2),
    'conv_5_2':        lambda C: ConvReLU(C, C // 2, 5, 2),
    'conv_6_2':        lambda C: ConvReLU(C, C // 2, 6, 2),
    'conv_7_2':        lambda C: ConvReLU(C, C // 2, 7, 2),
    'conv_5_3':        lambda C: ConvReLU(C, C // 2, 5, 3),
    'conv_7_3':        lambda C: ConvReLU(C, C // 2, 7, 3),
    'avg_pool_3':      lambda C: nn.AvgPool1d(kernel_size=3, stride=2, padding=1, count_include_pad=False),
    'avg_pool_5':      lambda C: nn.AvgPool1d(kernel_size=5, stride=2, padding=2, count_include_pad=False),
    'avg_pool_7':      lambda C: nn.AvgPool1d(kernel_size=7, stride=2, padding=3, count_include_pad=False),
    'max_pool_3':      lambda C: nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    'max_pool_5':      lambda C: nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
    'max_pool_7':      lambda C: nn.MaxPool1d(kernel_size=7, stride=2, padding=3)
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

        # 对卷积层进行权重初始化
        if hasattr(self.op[0], 'weight'):
            init.kaiming_uniform_(self.op[0].weight, nonlinearity='relu')
        if hasattr(self.op[0], 'bias') and self.op[0].bias is not None:
            init.zeros_(self.op[0].bias)

    def forward(self, x):
        return self.op(x)


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
    return x[:,:,::self.stride,::self.stride].mul(0.)
