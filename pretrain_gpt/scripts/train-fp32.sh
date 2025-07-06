#!/bin/bash
set -e

CHECKPOINT_PATH="pretrain_gpt/.ckpts/gpt-c4-fp32"
TENSORBOARD_LOGS_PATH="pretrain_gpt/.logs/gpt-c4-fp32"
VOCAB_FILE="pretrain_gpt/tokenizer/gpt2_vocab.json"
MERGE_FILE="pretrain_gpt/tokenizer/gpt2_merges.txt"
DATA_PATH="pretrain_gpt/data/c4_text_document"

# transfer dataset to faster local scratch
if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    echo "[$(hostname)] Transferring dataset..."
    cp -rp pretrain_gpt/data ${T4TMPDIR} 
    DATA_PATH="${T4TMPDIR}/data/c4_text_document"
    echo "[$(hostname)] Transfer complete"
fi

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 64
    --global-batch-size 1536
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then

    # TSUBAME4

    CACHE_PATH="pretrain_gpt/data/c4_text_document/cache"
    DATA_ARGS+=(--data-cache-path "$CACHE_PATH")

    DISTRIBUTED_ARGS=(
        --nproc-per-node $N_GPUS
        --nnodes $OMPI_COMM_WORLD_SIZE
        --rdzv-id $JOB_ID
        --rdzv-backend c10d
        --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT
    )

else

    # general GPU machine

    DISTRIBUTED_ARGS=(
        --nproc-per-node $N_GPUS
        --nnodes 1
        --master-addr localhost
        --master-port 12345
    )

fi

torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]}
