#!/bin/bash
# shellcheck disable=SC2068
# 读取参数
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

# 分离参数
run_dirs=${args[0]}
task_name=${args[1]}
fl_algorithm=${args[2]}
port=${args[3]}

# 读取 GPU 参数
idx=0
for(( i=4;i<${#args[@]};i++ ))
do
  device[${idx}]=${args[i]}
  let "idx=${idx}+1"
done

world_size=${#device[@]}
echo "world_size is ${world_size}"

if [ ${task_name} == "qnli" ];
then
  max_seq=256
else
  max_seq=128
fi
echo "${task_name}'s max_seq is ${max_seq}"

CUDA_VISIBLE_DEVICES=${device[0]} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/roberta-base/ \
--output_dir ${run_dirs}/output/fedglue \
--rank 0 \
--task_name ${task_name} \
--fl_algorithm ${fl_algorithm} \
--raw_dataset_path ${run_dirs}/data/fedglue \
--partition_dataset_path ${run_dirs}/data/fedglue \
--max_seq_length ${max_seq} \
--world_size ${world_size} \
--port ${port} &

sleep 2s

for(( i=1;i<${world_size};i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=${device[i]} python main.py \
    --model_name_or_path ${run_dirs}/pretrain/nlp/roberta-base/ \
    --output_dir ${run_dirs}/output/fedglue \
    --rank ${i} \
    --task_name ${task_name} \
    --fl_algorithm ${fl_algorithm} \
    --raw_dataset_path ${run_dirs}/data/fedglue \
    --partition_dataset_path ${run_dirs}/data/fedglue \
    --max_seq_length ${max_seq} \
    --world_size ${world_size} \
    --port ${port} &
    sleep 2s
}
done

wait
