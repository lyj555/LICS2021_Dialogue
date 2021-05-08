#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-05-04
# Brief:
#   LICS Dialogue部分 predict
# Arguments:
#     $1: model_path, required, the pretrained model path
#     $2: device, optional, cpu or gpu, default is gpu
#     $3: predict_data_path, optional, default is ./dataset/test.json
#     $4: python_path, optional, if empty, will use default python
# Example: bash run_eval.sh
# Returns:
#   succ: exit 0
#   fail: exit 1

model_path=$1
device=$2
predict_data_path=$3
python_path=$4

sh_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)  # current directory path
project_dir=${sh_dir}/..

# [1]. check params
[[ ! -d $model_path ]] && echo "model_path: $model_path is not valid directory" && exit 1
[[ -z $device ]] && device="gpu"
[[ -z $test_data_path ]] && test_data_path=${project_dir}/dataset/output/test.txt
[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

pretrained_model_path=$model_path
model_name=$(basename $pretrained_model_path)
output_dir=${project_dir}/output/predict/$(date +%m%d_%H%M)_model_name_$model_name  # output directory
mkdir -p $output_dir

batch_size=2
model_type="uniLM"

# [2]. do predict
${python_path} ${project_dir}/run_dialogue.py \
    --do_train 0 --do_eval 0 \
    --do_predict 1 --test_data_path $predict_data_path \
    --output_dir $output_dir \
    --device $device --model_type $model_type \
    --pretrained_model_path $pretrained_model_path \
    --batch_size $batch_size \
