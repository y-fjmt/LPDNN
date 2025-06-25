#!/bin/bash
#$-cwd
#$-l node_h=2
#$-l h_rt=00:20:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    source ~/.bash_profile
    apptainer \
        exec \
        --nv \
        -B .:/workspace \
        -B /var/spool/age \
        apptainer/pytorch.sif \
        ./pretrain_gpt/scripts/train-bf16.sh
else
    apptainer \
        exec \
        --nv \
        -B .:/workspace \
        apptainer/pytorch.sif \
        ./pretrain_gpt/scripts/train-bf16.sh
fi
