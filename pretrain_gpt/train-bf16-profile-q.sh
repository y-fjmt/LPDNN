#!/bin/bash
#$-cwd
#$-l node_q=1
#$-l h_rt=04:00:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then
    source ~/.bash_profile
fi

echo "copy start..."
cp -rp pretrain_gpt/data ${T4TMPDIR} 
echo "copy done."

apptainer \
    exec \
    --nv \
    -B .:/workspace \
    -B ${T4TMPDIR} \
    apptainer/pytorch.sif \
    ./pretrain_gpt/scripts/train-bf16-profile-q.sh


apptainer \
    exec \
    --nv \
    -B .:/workspace \
    -B ${T4TMPDIR} \
    apptainer/pytorch.sif \
    ./pretrain_gpt/scripts/train-bf16-single-q.sh
