#!/bin/bash
#$-cwd
#$-l cpu_40=1
#$-l h_rt=2:00:00

set -e

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    source ~/.bash_profile
fi

# create base apptainer image
apptainer build \
	--fakeroot \
	apptainer/ngc_pytorch_2505.sif \
	docker://nvcr.io/nvidia/pytorch:25.05-py3

# build from pytorch.def
apptainer build \
	apptainer/pytorch.sif \
	apptainer/pytorch.def
