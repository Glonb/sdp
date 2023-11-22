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
    # 'skip_connect',
    'conv_3_1',
    'conv_4_1',
    'conv_5_1',
    'conv_6_1',
    'conv_7_1',
    'conv_9_1',
    'conv_3_2',
    'conv_5_2',
    'conv_7_2',
    'conv_9_2',
    'conv_7_3',
    'conv_9_3'
    # 'sep_conv_3_1',
    # 'sep_conv_5_1',
    # 'sep_conv_7_1',
    # 'sep_conv_3_2',
    # 'sep_conv_5_2',
    # 'sep_conv_7_2',
    # 'dil_conv_3_1',
    # 'dil_conv_5_1',
    # 'dil_conv_7_1'
]

# SDP_Genotype = Genotype(geno=[('conv_6_2', 0), ('conv_6_2', 0), ('max_pool_7', 1), ('conv_6_1', 0)], geno_concat=range(1, 5))
# SDP = SDP_Genotype

# 初始化 SDP_Genotype
SDP_Genotype = Genotype(geno=[], geno_concat=[])

def set_Genotype(new_genotype):
    global SDP_Genotype
    SDP_Genotype = new_genotype
    
    # 将 genotype 写入文件
    write_genotype_to_file(new_genotype)

def get_Genotype():
    # 从文件中读取最后一行，并转换为 Genotype 对象
    return read_last_genotype_from_file()

def write_genotype_to_file(genotype):
    # 将 genotype 写入文件
    with open("genotype_history.txt", "a") as file:
        file.write(str(genotype) + "\n")

def read_last_genotype_from_file():
    # 从文件中读取最后一行
    try:
        with open("genotype_history.txt", "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()  # 去除末尾的换行符等空白字符
            # 将字符串转换为 Genotype 对象
            last_genotype = eval(last_line)
            return last_genotype
    except (IOError, IndexError, SyntaxError):
        # 处理文件读取失败、索引错误或语法错误
        return None
