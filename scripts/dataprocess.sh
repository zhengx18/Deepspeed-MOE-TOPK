#! /bin/bash

### step1 data process
# llama2 model

# BASE_PATH=/nlp_group/zhengxue/LLM/Pretrain/workspace
BASE_PATH=/nlp_group/liupeng15/pile

OUTPUT_PATH=/nlp_group/zhengxue/LLM/Pretrain/workspace/data/The_Pile

WORKERS=120
# 循环从00到29
for i in $(seq -w 00 29); do
# 构造并打印文件路径
inputfile=${BASE_PATH}/${i}.jsonl
output_prefix=${OUTPUT_PATH}/pile_gpt2_train_${i}
echo $inputfile
echo $output_prefix
echo "${output_prefix}.bin"
# 检查文件是否存在
if [ ! -f "${output_prefix}.bin" ]; then
# 文件不存在，执行 Python 脚本
echo 'file not exist'
python tools/preprocess_data.py \
       --input $inputfile \
       --output-prefix $output_prefix \
       --vocab-file /nlp_group/zhengxue/LLM/Pretrain/Codes/Megatron-DeepSpeed/dataset/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /nlp_group/zhengxue/LLM/Pretrain/Codes/Megatron-DeepSpeed/dataset/gpt2-merges.txt \
       --append-eod \
       --workers $WORKERS 
else
echo "file exist"
fi
done




