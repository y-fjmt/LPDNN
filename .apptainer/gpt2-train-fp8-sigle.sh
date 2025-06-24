#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=2:00:00

apptainer \
        exec \
        --nv \
        -B .:/workspace \
        -B /var/spool/age \
        .apptainer/pytorch.sif \
         bash -c 'TSUBAME_VERSION="disable" ./.apptainer/scripts/gpt2-train-fp8.sh'