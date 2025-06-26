#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=00:20:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    source ~/.bash_profile
fi

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    apptainer/pytorch.sif \
    ./pretrain_gpt/scripts/train-bf16.sh
