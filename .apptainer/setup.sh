#!/bin/bash
#$-cwd
#$-l cpu_40=1
#$-l h_rt=2:00:00

set -e

# create base apptainer image
apptainer build \
        .apptainer/ngc_pytorch_128.sif \
        docker://nvcr.io/nvidia/pytorch:25.05-py3

apptainer build \
        --fakeroot \
        .apptainer/pytorch.sif \
        .apptainer/pytorch.def
