#!/bin/bash
#$-cwd
#$-l node_h=2
#$-l h_rt=72:00:00

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    -B $HOME/.cache/huggingface:/root/.cache/huggingface \
    .apptainer/pytorch.sif \
    ./.apptainer/scripts/gpt2-train.sh