#!/bin/bash
#$-cwd
#$-l node_h=2
#$-l h_rt=1:00:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then

    # TSUBAME4

    source ~/.bash_profile

    module purge
    module load cuda/12.8.0
    module load openmpi/5.0.7-gcc

    cat $PE_HOSTFILE

    N_NODES=$(cat $PE_HOSTFILE | wc -l)

    MASTER_ADDR=$(hostname -i)
    MASTER_PORT=12345

    mpirun -np $N_NODES \
        --npernode 1 \
        -x HOSTNAME \
        -x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
        apptainer \
            exec \
            --nv \
            --bind .:/workspace \
            --bind ${T4TMPDIR} \
            --bind /var/spool/age \
            apptainer/pytorch.sif \
            ./pretrain_gpt/scripts/train-fp8.sh
else

    # general GPU machine
    apptainer \
        exec \
        --nv \
        --bind .:/workspace \
        apptainer/pytorch.sif \
        ./pretrain_gpt/scripts/train-fp8.sh
fi
