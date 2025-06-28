#!/bin/bash
#$-cwd
#$-l gpu_1=1
#$-l h_rt=2:00:00


if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    module purge
    module load cuda/12.8.0
    module load cudnn/9.8.0
    source ~/.bash_profile
fi

python3 -m venv .venv

.venv/bin/python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

.venv/bin/python3 -m \
    pip install -r apptainer/requirements.txt

# Megatron-LM
.venv/bin/python3 -m \
    pip install -e ./Megatron-LM

# Transformer Engine
.venv/bin/python3 -m \
pip install \
    --no-build-isolation transformer_engine[pytorch]

# NVIDIA APEX
git clone https://github.com/NVIDIA/apex
MAX_JOBS=$(nproc) \
NVCC_APPEND_FLAGS="--threads 4" \
.venv/bin/python3 -m \
pip install -v \
    --disable-pip-version-check \
    --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    ./apex