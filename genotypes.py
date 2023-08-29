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
    'smp_conv_3',
    'smp_conv_5',
    'smp_conv_7',
    'sep_conv_3',
    'sep_conv_5',
    'sep_conv_7',
    'dil_conv_3',
    'dil_conv_5'
]

SDP_Genotype = Genotype(
    geno=[('smp_conv_5', 0),
          ('dil_conv_5', 1), 
          ('smp_conv_5', 1), 
          ('sep_conv_5', 2), 
          ('skip_connect', 0), 
          ('sep_conv_5', 3)], 
    geno_concat=range(1, 7)
)
DARTS = SDP_Genotype
