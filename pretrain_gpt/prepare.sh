#!/bin/bash
#$-cwd
#$-l cpu_80=1
#$-l h_rt=24:00:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    source ~/.bash_profile
fi

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    apptainer/pytorch.sif \
    ./pretrain_gpt/scripts/prepare.sh
