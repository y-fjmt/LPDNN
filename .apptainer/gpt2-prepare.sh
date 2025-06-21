#!/bin/bash
#$-cwd
#$-l cpu_40=1
#$-l h_rt=24:00:00

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    .apptainer/pytorch.sif \
    ./.apptainer/scripts/gpt2-prepare.sh
