from    collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# 可选操作
PRIMITIVES = [
    'none',
    'max_pool_3',
    'max_pool_5',
    'max_pool_7',
    'avg_pool_3',
    'avg_pool_5',
    'avg_pool_7',
    'skip_connect',
    'sep_conv_3',
    'sep_conv_5',
    'sep_conv_7'
    # 'dil_conv_3',
    # 'dil_conv_5'
]

SDP_Genotype = Genotype(
    normal=[('sep_conv_5', 1), ('sep_conv_5', 0), 
            ('sep_conv_3', 2), ('sep_conv_3', 0), 
            ('sep_conv_7', 3), ('sep_conv_3', 2), 
            ('sep_conv_3', 3), ('sep_conv_3', 4)], 
    normal_concat=range(2, 6), 
    reduce=[('sep_conv_5', 1), ('sep_conv_3', 0), 
            ('max_pool_3', 2), ('sep_conv_7', 1), 
            ('sep_conv_7', 1), ('max_pool_3', 2), 
            ('skip_connect', 1), ('max_pool_3', 2)], 
    reduce_concat=range(2, 6)
)


DARTS = SDP_Genotype
