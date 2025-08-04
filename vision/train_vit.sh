#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=00:20:00

apptainer \
    exec \
    --nv \
    --bind .:/workspace \
    apptainer/pytorch.sif \
    bash -c "python3 vision/ViT/main.py --compute-dtype=fp16 --tensorboard-logdir=vision/.logs/fp16-$(date +%s)"