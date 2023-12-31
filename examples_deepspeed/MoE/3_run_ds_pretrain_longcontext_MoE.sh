#!/bin/bash
'''
10.82.120.21 slots=8
10.82.122.19 slots=8
10.82.122.81 slots=8
10.82.120.159 slots=8
10.82.120.215 slots=8
10.82.121.157 slots=8
'''
DIR=`pwd`
###############################################################################
RANK=$1
TOP_K=${TOP_K:-1}
MODEL_SIZE=${MODEL_SIZE:-125M}
DATA_PERCENT=${DATA_PERCENT:-0.5}
DEBUG=${DEBUG:-false}
RDMA=${RDMA:-true}
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=4096
MODEL_SIZE=1.3B
DATA_PERCENT=0.5
TOP_K=2
DEBUG=false
RDMA=true
ACTIVATION_CHECKPOINT=true 
WORLD_SIZE=8
Experts_Num=8
# MASTER_ADDR=11.38.199.29 # snowy_test开发机
MASTER_ADDR=10.82.124.33
MASTER_PORT=2333
KUBERNETES_CONTAINER_RESOURCE_GPU=8
echo 'DEBUG, RDMA, TOP_K, DATA_PERCENT, ACTIVATION_CHECKPOINT' $DEBUG $RDMA $TOP_K $DATA_PERCENT $ACTIVATION_CHECKPOINT

if [ "${DEBUG}" = "true" ] || [ "${RDMA}" = "false" ]; then
    export KUBERNETES_CONTAINER_RESOURCE_GPU=8
    export WORLD_SIZE=1
    export RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=2333
    RDMA="false"
else
    ## RDMA Configurations
    export NCCL_IB_HCA=mlx5_0
    export NCCL_IB_TC=136
    export NCCL_IB_SL=5
    export NCCL_IB_GID_INDEX=3
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_NET_PLUGIN=none
    export NCCL_DEBUG=INFO
    # export NCCL_P2P_DISABLE=1
    export NCCL_P2P_PXN_LEVEL=1
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=2
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
fi

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

## GPT-3 Small 125M
if [ "${MODEL_SIZE}" = "125M" ]; then
    MODEL_SIZE=125M
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTN_HEADS=12
    GLOBAL_BATCH_SIZE=1024
    ## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
    ## found that lower LR and min LR (than the base dense model) helps.
    ## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
    ## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
    ## heavily tuned.
    # LR=4.5e-4
    # MIN_LR=4.5e-06
    LR=9e-4
    MIN_LR=9e-06
    ## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
    BATCH_SIZE=8
    MP_SIZE=1
    ## Number of experts. EP_SIZE 1 means dense model without MoE
    EP_SIZE=64
    EP_PARALLEL_SIZE=1
elif [ "${MODEL_SIZE}" = "350M" ]; then
    ## GPT-3 Medium 350M
    MODEL_SIZE=350M
    NUM_LAYERS=24
    HIDDEN_SIZE=256
    NUM_ATTN_HEADS=16
    GLOBAL_BATCH_SIZE=120
    LR=3.0e-4
    MIN_LR=3.0e-5
    ## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
    BATCH_SIZE=5
    MP_SIZE=1
    ## Number of experts. EP_SIZE 1 means dense model without MoE
    EP_SIZE=32
    EP_PARALLEL_SIZE=8
elif [ "${MODEL_SIZE}" = "760M" ]; then
    # GPT-3 Large 760M
    MODEL_SIZE=760M
    NUM_LAYERS=24
    HIDDEN_SIZE=1536
    NUM_ATTN_HEADS=16
    GLOBAL_BATCH_SIZE=128
    LR=2.5e-4
    MIN_LR=2.5e-5
    ## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
    BATCH_SIZE=8
    MP_SIZE=2
    ## Number of experts. EP_SIZE 1 means dense model without MoE
    EP_SIZE=16
    EP_PARALLEL_SIZE=8
elif [ "${MODEL_SIZE}" = "1.3B" ]; then
    ## GPT-3 XL 1.3B
    MODEL_SIZE=1.3B
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    GLOBAL_BATCH_SIZE=1024
    LR=4.0e-4
    MIN_LR=4.0e-5
    ## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
    BATCH_SIZE=4
    MP_SIZE=1
    ## Number of experts. EP_SIZE 1 means dense model without MoE
    EP_SIZE=$Experts_Num # num-experts
    EP_PARALLEL_SIZE=8 # there will be num-experts/moe-expert-parallel-size experts on each GPU
elif [ "${MODEL_SIZE}" = "2.7B" ]; then
    ## GPT-3 2.7B
    MODEL_SIZE=2.7B
    NUM_LAYERS=32
    HIDDEN_SIZE=2560
    NUM_ATTN_HEADS=32
    GLOBAL_BATCH_SIZE=512
    LR=1.6e-4
    MIN_LR=1.6e-5
    ## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
    BATCH_SIZE=4
    MP_SIZE=4
    ## Number of experts. EP_SIZE 1 means dense model without MoE
    EP_SIZE=16
    EP_PARALLEL_SIZE=8
elif [ "${MODEL_SIZE}" = "6.7B" ]; then
    ## GPT-3 6.7B
    MODEL_SIZE=6.7B
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    GLOBAL_BATCH_SIZE=1024
    LR=1.2e-4
    MIN_LR=1.2e-5
elif [ "${MODEL_SIZE}" = "13B" ]; then
    ## GPT-3 13B
    MODEL_SIZE=13B
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    GLOBAL_BATCH_SIZE=1024
    LR=1.0e-4
    MIN_LR=1.0e-5
elif [ "${MODEL_SIZE}" = "175B" ]; then
    ## GPT-3 175B
    MODEL_SIZE=175B
    NUM_LAYERS=96
    HIDDEN_SIZE=12288
    NUM_ATTN_HEADS=96
    GLOBAL_BATCH_SIZE=1536
    LR=0.6e-4
    MIN_LR=0.6e-5
fi
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
if [ "${DEBUG}" = "true" ]; then
    TRAIN_TOKENS=1000000000
else
    TRAIN_TOKENS=$(echo "100000000000 * $DATA_PERCENT" | bc)
    TRAIN_TOKENS=$(printf "%.0f" $TRAIN_TOKENS)
    # TRAIN_TOKENS=330000000000
fi

## TRAIN_ITERS is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
if [ "${DEBUG}" = "true" ]; then
    WARMUP_TOKENS=1250000
    LR_DECAY_TOKENS=1000000000               
else
    WARMUP_TOKENS=$(echo "125000000 * $DATA_PERCENT" | bc)
    WARMUP_TOKENS=$(printf "%.0f" $WARMUP_TOKENS)
    LR_DECAY_TOKENS=$(echo "100000000000 * $DATA_PERCENT" | bc)
    LR_DECAY_TOKENS=$(printf "%.0f" $LR_DECAY_TOKENS)
fi
###############################################################################


## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
set -x
# NUM_GPUS=$(($(--format=csv,noheader | wc -l)-2))
# NUM_GPUS_PERNODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# NUM_NODE=$(( ${NUM_GPUS} / ${NUM_GPUS_PERNODE} ))
NUM_NODE=$WORLD_SIZE
NUM_GPUS_PERNODE=$KUBERNETES_CONTAINER_RESOURCE_GPU
NUM_GPUS=$(( ${NUM_NODE} * ${NUM_GPUS_PERNODE} ))
###############################################################################
## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs

if [ "${DEBUG}" = "true" ] || [ "${RDMA}" = "false" ]; then
    LOG_INTERVAL=1
else
    LOG_INTERVAL=10
fi
EVAL_ITERS=10
EVAL_INTERVAL=1000
SAVE_INTERVAL=5000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
# ACTIVATION_CHECKPOINT="true"
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-"false"}
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}-top${TOP_K}"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH="/mmu_nlp_hdd/zhengxue/Megatron-DeepSpeed/output0102_tp${TOP_K}_ep${EP_SIZE}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR} 
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
DATA_HOME=/nlp_group/zhengxue/LLM/Pretrain/workspace/data/The_Pile
VOCAB_PATH=/nlp_group/zhengxue/LLM/Pretrain/Codes/Megatron-DeepSpeed/dataset/gpt2-vocab.json
MERGE_PATH=/nlp_group/zhengxue/LLM/Pretrain/Codes/Megatron-DeepSpeed/dataset/gpt2-merges.txt
if [ "${DEBUG}" = "true" ]; then
    # VOCAB_PATH=/cpfs/2926428ee2463e44/user/yuxiaoqing/repos/moe/Megatron-DeepSpeed/vocab/gpt2-vocab.json
    # MERGE_PATH=/cpfs/2926428ee2463e44/user/yuxiaoqing/repos/moe/Megatron-DeepSpeed/vocab/gpt2-merges.txt
    # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    # For cluster Azure-EastUS-V100-32GB-4, Lab-RR1-V100
    # DATA_HOME=/data/nas/chengfeng2/data/public/the_pile/The_Pile/processed4gpt2/train
    # For cluster Azure-WestUS3-A100
    # DATA_PATH=/blob/data/the_pile_public_merged_nopreprocessing/pile_text_document
    DATA_PATH=" 1 ${DATA_HOME}/pile_gpt2_train_00_text_document"
else
    # VOCAB_PATH=/cpfs/2926428ee2463e44/user/yuxiaoqing/repos/moe/Megatron-DeepSpeed/vocab/gpt2-vocab.json
    # MERGE_PATH=/cpfs/2926428ee2463e44/user/yuxiaoqing/repos/moe/Megatron-DeepSpeed/vocab/gpt2-merges.txt
    # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    # For cluster Azure-EastUS-V100-32GB-4, Lab-RR1-V100
    # DATA_HOME=/data/nas/chengfeng2/data/public/the_pile/The_Pile/processed4gpt2/train
    # For cluster Azure-WestUS3-A100
    # DATA_PATH=/blob/data/the_pile_public_merged_nopreprocessing/pile_text_document
    DATA_PATH=" 0.03 ${DATA_HOME}/pile_gpt2_train_00_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_01_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_02_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_03_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_04_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_05_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_06_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_07_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_08_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_09_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_10_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_11_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_12_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_13_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_14_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_15_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_16_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_17_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_18_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_19_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_20_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_21_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_22_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_23_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_24_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_25_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_26_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_27_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_28_text_document \
                0.03 ${DATA_HOME}/pile_gpt2_train_29_text_document"
fi


###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"
        
megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --topk ${TOP_K} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 8 \
        --bf16 \
        --overlap-p2p-communication \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

template_json="ds_config_gpt_TEMPLATE.json"
config_json="ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
	  > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
ITERATION_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
ITERATION_FILE_2="$CHECKPOINT_PATH/latest"
ITERATION=0
for (( node = 0; node <= NUM_NODE-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$ITERATION_FILE\""); then
        LOCAL_ITERATION=$(ssh -q worker-"$node" cat $ITERATION_FILE)
        ITERATION=$(( ${LOCAL_ITERATION} > ${ITERATION} ? ${LOCAL_ITERATION} :  ${ITERATION} ))
    fi
done
if [[ $ITERATION -gt 0 ]]; then
    ITERATION_2="global_step${ITERATION}"
    ds_ssh "echo $ITERATION > $ITERATION_FILE"
    ds_ssh "echo $ITERATION_2 > $ITERATION_FILE_2"
fi
set -x
if [ "${DEBUG}" = "true" ] || [ "${RDMA}" = "false" ]; then
    torchrun --nproc_per_node $KUBERNETES_CONTAINER_RESOURCE_GPU --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} 2>&1 | tee -a debug.log
else
    # run_cmd="deepspeed ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
    run_cmd="torchrun --nproc_per_node $KUBERNETES_CONTAINER_RESOURCE_GPU --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
    # run_cmd="torchrun --nproc_per_node $KUBERNETES_CONTAINER_RESOURCE_GPU --nnodes $WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
    # run_cmd="torchrun --nproc_per_node $KUBERNETES_CONTAINER_RESOURCE_GPU --nnodes $WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} "
    echo ${run_cmd}
    eval ${run_cmd}    
fi
