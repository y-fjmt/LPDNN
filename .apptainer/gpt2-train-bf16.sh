#!/bin/bash
#$-cwd
#$-l node_h=2
#$-l h_rt=5:00:00

if [ TSUBAME_VERSION = "4.0" ]; then
    apptainer \
        exec \
        --nv \
        -B .:/workspace \
        -B /var/spool/age \
        .apptainer/pytorch.sif \
        ./.apptainer/scripts/gpt2-train-bf16.sh
else
    apptainer \
        exec \
        --nv \
        -B .:/workspace \
        .apptainer/pytorch.sif \
        ./.apptainer/scripts/gpt2-train-bf16.sh
fi
