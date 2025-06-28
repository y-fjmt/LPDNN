#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=00:30:00

set -e

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    module purge
    module load cuda/12.8.0
    module load cudnn/9.8.0
    source ~/.bash_profile
fi

CHECKPOINT_PATH="pretrain_gpt/.ckpts/gpt-c4-bf16"
TENSORBOARD_LOGS_PATH="pretrain_gpt/.logs/gpt-c4-bf16"
VOCAB_FILE="pretrain_gpt/tokenizer/gpt2_vocab.json"
MERGE_FILE="pretrain_gpt/tokenizer/gpt2_merges.txt"
DATA_PATH="pretrain_gpt/data/c4_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port 12345
)

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
    --train-iters 10 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
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
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

nsys profile .venv/bin/python3 -m \
torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
