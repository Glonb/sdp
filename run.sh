#!/bin/bash

# projects=('ant', 'camel', 'ivy', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces')

declare -A versions=(
    ['ant']='1.5 1.6 1.7'
    ['camel']='1.2 1.4 1.6'
    ['ivy']='1.4 2.0'
    ['jedit']='3.2 4.0 4.1'
    ['log4j']='1.0 1.1'
    ['lucene']='2.0 2.2 2.4'
    ['poi']='1.5 2.5 3.0'
    ['synapse']='1.0 1.1 1.2'
    # ['velocity']='1.4 1.5 1.6'
    ['xalan']='2.4 2.5 2.6'
    ['xerces']='1.1 1.2 1.3'
)

# declare -A versions=(
#     ['ant']='1.3 1.4 1.5 1.6 1.7'
#     ['camel']='1.2 1.4 1.6'
#     ['ivy']='1.4 2.0'
#     ['jedit']='3.2 4.0 4.1 4.2'
#     ['log4j']='1.0 1.1'
#     ['lucene']='2.0 2.2 2.4'
#     ['poi']='1.5 2.0 2.5 3.0'
#     ['synapse']='1.0 1.1 1.2'
#     ['velocity']='1.4 1.5 1.6'
#     ['xalan']='2.4 2.5 2.6'
#     ['xerces']='1.1 1.2 1.3'
# )

# 遍历关联数组的键
for key in "${!versions[@]}"; do
    # 获取关联数组中的值
    values=${versions[$key]}
    
    # 将值分割为数组
    values_array=($values)
    
    # 遍历值数组，每次取相邻的两个值
    for ((i=0; i<${#values_array[@]}-1; i++)); do
        value1=${values_array[$i]}
        value2=${values_array[$i+1]}
        echo "________project ${key}_${value1}_${value2} start!________"
        # python gh-lstm.py --train_data "$key-$value1" --test_data "$key-$value2" --batchsz 256
        python dp-cnn.py --train_data "$key-$value1" --test_data "$key-$value2"
        # python train_search.py --data "$key-$value1" --epochs 20 --layers 4
        # echo "search finished!"
        # python train.py --data "$key-$value1" --epochs 1 --layers 4
        # python test.py --data "$key-$value2" --layers 4
    done
done
