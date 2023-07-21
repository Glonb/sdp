from    collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# 可选操作
PRIMITIVES = [
    'none',
    'max_pool_3',
    'avg_pool_3',
    'skip_connect',
    'sep_conv_3',
    'sep_conv_5',
    'dil_conv_3',
    'dil_conv_5'
]

DARTS_MINE = Genotype(
    normal=[('sep_conv_3', 1), ('dil_conv_3', 0), 
            ('sep_conv_5', 2), ('sep_conv_5', 1),
            ('dil_conv_5', 2), ('dil_conv_3', 1), 
            ('dil_conv_3', 3), ('dil_conv_5', 1)],
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3', 0), ('sep_conv_3', 1), 
            ('avg_pool_3', 0), ('dil_conv_3', 1), 
            ('avg_pool_3', 0), ('dil_conv_3', 3), 
            ('sep_conv_3', 0), ('dil_conv_5', 3)], 
    reduce_concat=range(2, 6)
)

DARTS = DARTS_MINE
