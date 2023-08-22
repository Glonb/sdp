from    collections import namedtuple


Genotype = namedtuple('Genotype', 'geno geno_concat')

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
    geno=[('avg_pool_5', 0), 
          ('sep_conv_7', 1), 
          ('sep_conv_5', 2), 
          ('sep_conv_5', 2), 
          ('sep_conv_7', 4), 
          ('sep_conv_5', 4)], 
    geno_concat=range(2, 8)
)

DARTS = SDP_Genotype
