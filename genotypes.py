from    collections import namedtuple


Genotype = namedtuple('Genotype', 'geno geno_concat')

# 可选操作
PRIMITIVES = [
    # 'max_pool_3',
    # 'max_pool_5',
    # 'max_pool_7',
    # 'avg_pool_3',
    # 'avg_pool_5',
    # 'avg_pool_7',
    # 'skip_connect',
    'conv_3_1',
    'conv_4_1',
    'conv_5_1',
    'conv_6_1',
    'conv_7_1',
    # 'conv_9_1',
    'conv_3_2',
    'conv_4_2',
    'conv_5_2',
    'conv_6_2',
    'conv_7_2',
    # 'conv_9_2',
    'sep_conv_3_1',
    'sep_conv_5_1',
    # 'sep_conv_7_1',
    'sep_conv_3_2',
    'sep_conv_5_2',
    # 'sep_conv_7_2',
    # 'dil_conv_3_1',
    # 'dil_conv_5_1',
    # 'dil_conv_7_1'
]

SDP_Genotype = Genotype(
    geno=[('conv_5_2', 0), 
          ('conv_7_2', 0), 
          ('conv_6_2', 0),
          ('conv_6_1', 0)],
    geno_concat=range(1, 5)
)
SDP = SDP_Genotype
