# 下载 huggingface 格式的 llama2 checkpoint
# mkdir -p /workspace/model_ckpts && cd /workspace/model_ckpts
# wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/Llama-2-7b-hf.tgz
# tar -zxf Llama-2-7b-hf.tgz
# mv Llama-2-7b-hf llama2-7b-hf

# 将 hf 格式的 checkpoint 转成 megatron 格式，这里需要提前指定 tp、pp 的 size
# cp -r /workspace/Megatron-LM/tools/checkpoint /workspace/model_ckpts 
# cd /workspace/model_ckpts && mv ./checkpoint ./ckpt_convert
# python ./ckpt_convert/util.py \
# --model-type GPT \
# --loader llama2_hf \
# --saver megatron \
# --target-tensor-parallel-size 2 \
# --target-pipeline-parallel-size 1 \
# --load-dir ./llama2-7b-hf \
# --save-dir ./llama2-7b_hf-to-meg_tp2-pp1 \
# --tokenizer-model ./llama2-7b-hf/tokenizer.model \
# --megatron-path /workspace/Megatron-LM


# 指定要遍历的文件夹路径
# folder_path="/path/to/your/folder"
OUTPUT_PATH=/nlp_group/zhengxue/LLM/Pretrain/workspace/data/The_Pile

# 遍历文件夹中的所有文件
for file in "$OUTPUT_PATH"/*; do
    # 检查文件名是否符合模式
    if [[ $file =~ pile_gpt2_train_[0-9]+_text_document_text_document.idx ]]; then
        # 构造新的文件名
        new_file=$(echo $file | sed 's/_text_document_text_document.idx/_text_document.idx/')
        # 重命名文件
        mv "$file" "$new_file"
        echo "Renamed $file to $new_file"
    fi
    if [[ $file =~ pile_gpt2_train_[0-9]+_text_document_text_document.bin ]]; then
        # 构造新的文件名
        new_file=$(echo $file | sed 's/_text_document_text_document.bin/_text_document.bin/')
        # 重命名文件
        mv "$file" "$new_file"
        echo "Renamed $file to $new_file"
    fi
done