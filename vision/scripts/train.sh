#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=24:00:00

source ~/.bash_profile

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    BINDING=(
        --bind .:/workspace
        --bind /gs/fs
    )
    IMAGENET_ROOT="/gs/fs/datasets/academic/ILSVRC2012"
else
    BINDING=(
        --bind .:/workspace
    )
    IMAGENET_ROOT="vision/data/imagenet"
fi

apptainer \
    exec \
    --nv \
    ${BINDING[@]} \
    apptainer/pytorch.sif \
    python3 vision/main.py \
        --model b16 \
        --dtype fp16 \
        --weight-dtype fp32 \
        --lr 5e-3 \
        --epoch 10 \
        --batch-size 4096 \
        --mini-batch-size 512 \
        --imagenet-root $IMAGENET_ROOT \
        --workers 8 \
        --tensorboard-logdir vision/.logs/fp16-$(date +%s)