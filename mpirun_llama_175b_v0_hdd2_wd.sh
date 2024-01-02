#!/bin/bash

ts=`date +%Y_%m_%d_%H_%M`

# TOKENIZER=/nlp_group/zxw/resources/tokenizers/spm.llama+64k.vocab=80496.prefer=old.remove-old-zh-piece=true.model
VOCAB_SIZE=128000

TP=8
PP=8
SP='--sequence-parallel'
HIDDEN_SIZE=12288
FFN_HIDDEN_SIZE=32768
LAYERS=96
NUM_ATTENTION_HEADS=96

SEQ_LENGTH=4096
MICRO_BATCH=1
GLOBAL_BATCH=960
GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    ${SP} \
    --use-distributed-optimizer \
    --optimizer adam \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1048576 \
    --lr-decay-iters 524288 \
    --lr-warmup-iters 15000 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --warmup-init-lr 1.e-7 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --start-weight-decay 0.01 \
    --end-weight-decay 0.1 \
    --weight-decay-incr-style linear \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1.e-8 \
    --bf16 \
    --use-flash-attn \
    --use-alibi \
    --seed 9527 \
    --no-load-rng \
    --init-method-std 0.02 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --overlap-p2p-communication \
    --override-opt_param-scheduler \
"
#    --min-lr 1.5e-5 \
#    --train-iters 524288 \
#    --lr-decay-iters 524288 \
#     --disable-output-scale-init \
#    --custom-lr-scheduler \
#    --weight-decay 0.01 \ before1208 425k change to 0.1

TRAIN_DATA_PATH='0.0012942246808510635	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__OpenSource_pile_Books3
                9.495319148936169e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__OpenSource_pile_BookCorpus2
                0.00018535387234042553	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__OpenSource_pile_Gutenberg
                0.00785900119148936	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__DATA_zlib
                5.85327659574468e-06	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__DATA_cbook
                0.007801768851063829	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__DATA_weixin_dushu
                0.002404396595744681	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__DATA_biqu_ge
                0.00035444834042553194	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Book__DATA_epubs
                0.0072223744690585925        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Baike__DATA_wiki__zh
                0.0012037292933093412        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Baike__DATA_wiki__en
                0.011573896237632067	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Baike__DATA_baidu_baike
                0.013681427999999997	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/News__DATA_chinese_news
                0.016318572	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/News__SELF_CC_NEWS
                0.5828571424	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/CC__SELF_CC__en
                0.09714285759	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/CC__SELF_CC_ALL__zh
                0.01714285566765989        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/WebText__DATA_webtext__zh
                0.0028571443323401075        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/WebText__DATA_webtext__en
                0.0027035324669688254	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/QA__SELF_stackexchange
                0.004522141175377637	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/QA__DATA_zhihu_qa
                0.0017250596117894868	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/QA__DATA_baidu_zhidao
                0.0005219244734164844	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/QA__DATA_baidu_jingyan
                0.0005273422724475657	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/QA__DATA_zhihu_wenzhang
                0.03000000032091566              /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Academic__DATA_baidu_xueshu__zh
                0.005000000628064525              /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Academic__DATA_baidu_xueshu__en
                0.06499999905101982	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Academic__DATA_arxiv
                0.05991135737	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Code__SELF_github_star1
                8.864262556662125e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Code__DATA_leetcode
                0.0018777263751366428	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_lyric
                5.9683406711159825e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_PhilPapers
                0.006454969797522676	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_FreeLaw
                0.00015150332538296413	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_DMMathematics
                0.00194200042033457	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_USPTOBackgrounds
                0.0007666999464416303	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_OpenSubtitles
                0.00016986792188330768	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__OpenSource_pile_NIHExPorter
                0.0022955081795614863	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Academic__OpenSource_pile_PubMedAbstracts
                0.009218761911246631	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_weixin_gongzhonghao
                2.2955132859446748e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_china_baogao_dating
                2.2955132859446748e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_1999IT
                9.182032718245944e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_yi_ou_zhi_ku
                1.8364085862024646e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_airuiwang
                0.00022495966883106272	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_36kr
                0.00025250554230494025	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_dongfang_caifu
                0.006119825613519466	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_xiao_hong_shu
                0.001143162940655657	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_dazhong_dianping
                3.213712472662719e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__Github_gushi
                0.00582600023972707	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_jianshu
                0.014756839931895352         /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_csdn__zh
                0.0024594735416927772         /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_csdn__en
                0.0014461706126982232	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_bokeyuan
                5.509215545841015e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_mafengwo_gonglv
                0.0019465910588213366	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_mafengwo_youji
                0.0004912391691495795	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_douban_subject
                0.0019006805888471155	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_douban_changping
                3.213712472662719e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_xia_chu_fang
                0.0001377303886460254	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_qiche_zhijia
                8.263833531527899e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Domain__DATA_boss_zhipin'

                # 2.142613790304677e-06	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.csharp128k_data_ratio1_complete.merge/Code__DATA_w3school

DATA_ARGS="
    --iterable-dataset \
    --train-data-path $TRAIN_DATA_PATH \
    --data-impl mmap \
    --tokenizer-type NullTokenizer \
    --vocab-size $VOCAB_SIZE \
"
    # --tokenizer-model $TOKENIZER \

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 0 \
    --timing-log-level 0  
"

STABILITY_ARGS="
    --loss_median_window 200 \
    --stability_protection \
    --anomaly_times 2 \
    --skip_steps 200 \
    --consecutive_anomalies_steps 10 \
    --restart_skip_counter 0
"

LLAMA_ARGS="
    --rms-norm \
	--swiglu \
	--prefetch-factor 256 \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --no-bias-dropout-fusion \
    --untie-embeddings-and-output-weights \
    --kaimm-async-dataloader \
    --disable-bias-linear \
    --no-position-embedding \
    --accumulate-allreduce-grads-in-fp32 \
    --kaimm-overlap-optimizer-communication \
    --kaimm-overlap-reduce-ratio 0.125 \
    --kaimm-overlap-gather-ratio 0.125 \
    --kaimm-overlap-optimizer-slow-ctas 2 \
    --no-masked-softmax-fusion \
    --use-fast-rms-norm 
"
    # --use-rotary-position-embeddings \
    # --use-fast-rope \


# YOUR_MEGATRON_PATH=/root/Megatron-LM
# YOUR_MEGATRON_PATH=/nlp_group/chenxiansheng/code/Megatron_Dir/Megatron-LM-175B
YOUR_MEGATRON_PATH=/nlp_group/chenxiansheng/code/Megatron_Dir/Megatron-LM-175B-AlibiAlign-Valid-wd
# YOUR_MEGATRON_PATH=/home/liuchenhao/Megatron-LM-dev/Megatron-LM
SAVE_PATH=/mmu_nlp_hdd/glb/Llama-hybrid-parallelism/llmbenchmark_175b/Llama/pretrain/175b_v0_hdd2_wd
EXPERIMENT_NAME=2kgpus-175b-scratch
RUN_NAME=175b_v0_hdd_continue2_wd

if [ ! -d "$SAVE_PATH" ]; then
  mkdir -p "$SAVE_PATH"
fi

CHECKPOINT_PATH=$SAVE_PATH/save

if [ ! -d "$CHECKPOINT_PATH" ]; then
  mkdir -p "$CHECKPOINT_PATH"
fi

LOG_PATH=$SAVE_PATH/log

if [ ! -d "$LOG_PATH" ]; then
  mkdir -p "$LOG_PATH"
fi


# cp $TOKENIZER $CHECKPOINT_PATH/tokenizer.model
# python ./config4eval/utils.py \
#   --max_length $SEQ_LENGTH \
#   --vocab_size $VOCAB_SIZE \
#   --hidden_size $HIDDEN_SIZE \
#   --intermediate_size $FFN_HIDDEN_SIZE \
#   --num_attention_heads $NUM_ATTENTION_HEADS \
#   --num_hidden_layers $LAYERS


cp ./config175B/config.json $CHECKPOINT_PATH
# cp ./config4eval/tokenizer_config.json $CHECKPOINT_PATH

hostfile=/etc/mpi/hostfile_seq
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
set -x
mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-F $TARGET_IP_PORT_FILE" \
    -hostfile $hostfile \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x PYTHONPATH=${YOUR_MEGATRON_PATH} \
        -bind-to none  -map-by slot \
	-mca plm_rsh_num_concurrent 300 \
	-mca routed_radix 600 \
	-mca btl_tcp_if_include eth04 \
    -mca btl_openib_allow_ib true \
    --mca btl tcp,self \
	-x NCCL_IB_QPS_PER_CONNECTION=2 \
	-x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_DEBUG=WARN \
    -x LD_LIBRARY_PATH -x PATH \
	python ./train_iterable_dsets_mlflow_llama_175b_v0_hdd_rm_wd_norm.py \
            --save_path ${SAVE_PATH} \
            --experiment_name ${EXPERIMENT_NAME} \
            --run_name ${RUN_NAME} \
            $GPT_ARGS \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            $LLAMA_ARGS \
            $STABILITY_ARGS \
            --distributed-backend nccl \
            --make-vocab-size-divisible-by 128 \
            --load ${CHECKPOINT_PATH} \
            --save ${CHECKPOINT_PATH} \
            --master-addr ${MY_NODE_IP}:8389 2>&1 | tee $LOG_PATH/llama_$ts.log
            
exit_code=${PIPESTATUS[0]}  # capture the exit code of mpirun
exit $exit_code
