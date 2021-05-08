#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-05-03
# Brief:
#   LICS Dialogue部分模型训练
# Arguments:
#     $1: train_data_path, optional, train data path
#     $2: valid_data_path, optional, dev data path
#     $3: device, optional, cpu or gpu, default is gpu
#     $4: python_path, optional, if empty, will use default python
# Example: bash run_train.sh <train data path>
# Returns:
#   succ: exit 0
#   fail: exit 1

train_data_path=$1
valid_data_path=$2
device=$3
python_path=$4

[[ -z $device ]] && device="cpu"
[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

sh_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)  # current directory path
project_dir=${sh_dir}/..

[[ -z $train_data_path ]] && train_data_path=${project_dir}/output/train.txt
[[ -z $valid_data_path ]] && valid_data_path=${project_dir}/output/dev.txt
output_dir=${project_dir}/output/train/$(date +%m%d_%H%M)  # output directory

model_type="uniLM"
pretrained_model_path="unified_transformer-12L-cn"
train_epochs=2
batch_size=8192

# optimizer(adamX)
lr=1e-5
weight_decay=0.01
warmup_steps=4000
max_grad_norm=0.1

# steps
logging_steps=1000
save_steps=100000


${python_path} ${project_dir}/run_dialogue.py \
    --do_train 1 --train_data_path $train_data_path \
    --do_eval 1 --valid_data_path $valid_data_path \
    --do_predict 0 \
    --output_dir $output_dir \
    --device $device --seed 666666 --model_type $model_type \
    --pretrained_model_path $pretrained_model_path \
    --train_epochs $train_epochs --batch_size $batch_size \
    --logging_steps $logging_steps --save_steps $save_steps \
    --lr $lr --weight_decay $weight_decay \
    --warmup_steps $warmup_steps --max_grad_norm $max_grad_norm \
    --logging_steps $logging_steps --save_steps $save_steps
