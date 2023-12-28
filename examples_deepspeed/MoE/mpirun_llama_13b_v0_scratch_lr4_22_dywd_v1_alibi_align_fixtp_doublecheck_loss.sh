#!/bin/bash

ts=`date +%Y_%m_%d_%H_%M`

TOKENIZER=/nlp_group/zxw/resources/tokenizers/spm.llama+64k.vocab=80496.prefer=old.remove-old-zh-piece=true.model
VOCAB_SIZE=80496

TP=2
PP=4
SP='--sequence-parallel' #  '--sequence-parallel' # '--sequence-parallel'
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824
LAYERS=40
NUM_ATTENTION_HEADS=40

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
    --train-iters 524288 \
    --lr-decay-iters 524288 \
    --lr-warmup-iters 15000 \
    --lr 4.e-4 \
    --min-lr 4.0e-5 \
    --warmup-init-lr 1.e-7 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --start-weight-decay 0.1 \
    --end-weight-decay 0.01 \
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
"
    # --custom-lr-scheduler \
    #     --recompute-granularity full \
    # --recompute-method block \


TRAIN_DATA_PATH='0.0029452993590268427	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__OpenSource_pile_Books3
                0.0002160873441597992	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__OpenSource_pile_BookCorpus2
                0.00042181442640887334	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__OpenSource_pile_Gutenberg
                0.0178849248622454	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__DATA_zlib
                1.3320447415913191e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__DATA_cbook
                0.017754679798876592	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__DATA_weixin_dushu
                0.005471745251864868	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__DATA_biqu_ge
                0.0008066269213603286	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Book__DATA_epubs
                0.011190562        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Baike__DATA_wiki__zh
                0.001865094        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Baike__DATA_wiki__en
                0.017932939365531204	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Baike__DATA_baidu_baike
                0.034839734763319655	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/News__DATA_chinese_news
                0.04155521778838692	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/News__SELF_CC_NEWS
                0.075066736	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/CC__SELF_CC__en
                0.484334693	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0622.merge/CC__SELF_CC__en_add
                0.012511123	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/CC__SELF_CC__zh
                0.080722449	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.cc.0608.merge/CC__SELF_CC__zh_2022add
                0.001660103        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/WebText__DATA_webtext__zh
                0.000276684        /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/WebText__DATA_webtext__en
                0.008377791358032021	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/QA__SELF_stackexchange
                0.01401335316729392	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/QA__DATA_zhihu_qa
                0.005345668929193105	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/QA__DATA_baidu_zhidao
                0.001617355957939179	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/QA__DATA_baidu_jingyan
                0.0016341448038127543	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/QA__DATA_zhihu_wenzhang
                0.01740406              /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Academic__DATA_baidu_xueshu__zh
                0.002900677              /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Academic__DATA_baidu_xueshu__en
                0.037708795712751625	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0622.merge/Academic__DATA_arxiv
                0.01882290101689326	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Code__SELF_github_star5
                0.039102782892005646	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0622.merge/Code__SELF_github_star1
                8.570469664601795e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Code__DATA_leetcode
                2.142613790304677e-06	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Code__DATA_w3school
                0.0014243961766912821	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_lyric
                4.527433680271996e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_PhilPapers
                0.004896578341761811	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_FreeLaw
                0.00011492662631199687	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_DMMathematics
                0.0014731528568192628	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_USPTOBackgrounds
                0.0005815993676402386	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_OpenSubtitles
                0.0001288575490427722	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_NIHExPorter
                0.0017413149848805889	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__OpenSource_pile_PubMedAbstracts
                0.006993121784983905	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_weixin_gongzhonghao
                1.7413188584549216e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_china_baogao_dating
                1.7413188584549216e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_1999IT
                6.965259939522355e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_yi_ou_zhi_ku
                1.3930535373342042e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_airuiwang
                0.00017064876780536502	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_36kr
                0.00019154437718666147	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_dongfang_caifu
                0.004642346361716394	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_xiao_hong_shu
                0.0008671747617576004	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_dazhong_dianping
                2.4378417535476906e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__Github_gushi
                0.00441945779574292	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_jianshu
                0.011194169         /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Domain__DATA_csdn__zh
                0.001865695         /nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0803_with_lang.merge/Domain__DATA_csdn__en
                0.0010970287890964608	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_bokeyuan
                4.179152864853946e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_mafengwo_gonglv
                0.0014766352001445231	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_mafengwo_youji
                0.00037264172439754125	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_douban_subject
                0.0014418086680324513	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_douban_changping
                2.4378417535476906e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_xia_chu_fang
                0.00010447882162134867	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_qiche_zhijia
                6.268737044429585e-05	/nlp_group/liupeng15/code/querysim/src/corpus_process/export/data.0606.merge/Domain__DATA_boss_zhipin'


DATA_ARGS="
    --iterable-dataset \
    --train-data-path $TRAIN_DATA_PATH \
    --data-impl mmap \
    --tokenizer-model $TOKENIZER \
"

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
"
# llama
LLAMA_ARGS="
    --rms-norm \
    --swiglu \
	--prefetch-factor 128 \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --untie-embeddings-and-output-weights \
    --kaimm-async-dataloader \
    --disable-bias-linear \
    --no-position-embedding \
    --accumulate-allreduce-grads-in-fp32 \
    --kaimm-overlap-optimizer-communication \
    --kaimm-overlap-reduce-ratio .19 \
    --kaimm-overlap-gather-ratio 0 
    --kaimm-overlap-optimizer-slow-ctas 8 \
    --no-masked-softmax-fusion \
    --use-fast-rms-norm \
    --use-alibi \
    --alibi-bias-max 8 \
"
# --use-rotary-position-embeddings \
# --use-fast-rope \
    # --no-bias-dropout-fusion \

# YOUR_MEGATRON_PATH=/root/Megatron-LM
YOUR_MEGATRON_PATH=/nlp_group/chenxiansheng/code/Megatron_Dir/Megatron-LM-175B-AlibiAlign-Valid
SAVE_PATH=./13b_v0_lr4_22_dywd_v1_alibi_align_fixedtp_doublecheck_dataloader
EXPERIMENT_NAME=2kgpus-13b-scratch-v1
RUN_NAME=2T-v0-lr4-22-dywd-v1-alibi-align-fixtp-doublecheck-dataloader

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


cp $TOKENIZER $CHECKPOINT_PATH/tokenizer.model
python ./config4eval/utils.py \
  --max_length $SEQ_LENGTH \
  --vocab_size $VOCAB_SIZE \
  --hidden_size $HIDDEN_SIZE \
  --intermediate_size $FFN_HIDDEN_SIZE \
  --num_attention_heads $NUM_ATTENTION_HEADS \
  --num_hidden_layers $LAYERS


cp ./config4eval/config.json $CHECKPOINT_PATH
cp ./config4eval/tokenizer_config.json $CHECKPOINT_PATH

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
        python ./train_iterable_dsets_mlflow_llama_13b.py \
            --save_path ${SAVE_PATH} \
            --experiment_name ${EXPERIMENT_NAME} \
            --run_name ${RUN_NAME} \
            $GPT_ARGS \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            $LLAMA_ARGS \
            $STABILITY_ARGS \
            --distributed-backend nccl \
            --make-vocab-size-divisible-by 1 \
            --load ${CHECKPOINT_PATH} \
            --save ${CHECKPOINT_PATH} \
            --master-addr ${MY_NODE_IP}:8389 2>&1 | tee $LOG_PATH/llama_$ts.log
            
exit_code=${PIPESTATUS[0]}  # capture the exit code of mpirun
exit $exit_code
