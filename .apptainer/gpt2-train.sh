#!/bin/bash
#$-cwd
#$-l node_h=2
#$-l h_rt=5:00:00


apptainer \
    exec \
    --nv \
    -B .:/workspace \
    -B /var/spool/age \
    .apptainer/pytorch.sif \
    ./.apptainer/scripts/gpt2-train-bf16.sh
