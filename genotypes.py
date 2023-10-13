from    collections import namedtuple


Genotype = namedtuple('Genotype', 'geno geno_concat')

# 可选操作
PRIMITIVES = [
    'max_pool_3',
    'max_pool_5',
    'max_pool_7',
    'avg_pool_3',
    'avg_pool_5',
    'avg_pool_7',
    'skip_connect',
    'conv_3',
    'conv_5',
    'conv_7',
    'sep_conv_3',
    'sep_conv_5',
    'sep_conv_7',
    'dil_conv_3',
    'dil_conv_5',
    'dil_conv_7'
]

SDP_Genotype = Genotype(geno=[('conv_7', 0), ('conv_7', 0), ('conv_7', 0), ('max_pool_3', 2), ('conv_7', 0), ('max_pool_5', 1)], geno_concat=range(1, 5))
SDP = SDP_Genotype
