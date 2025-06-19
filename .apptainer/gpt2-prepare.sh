#!/bin/bash
#$-cwd
#$-l cpu_80=1
#$-l h_rt=72:00:00

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    -B $HOME/.cache/huggingface:/root/.cache/huggingface \
    .apptainer/pytorch.sif \
    ./.apptainer/scripts/gpt2-prepare.sh