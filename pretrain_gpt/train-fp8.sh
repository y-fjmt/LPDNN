#!/bin/bash
#$-cwd
#$-l node_q=2
#$-l h_rt=01:00:00

if [ "$SGE_CLUSTER_NAME" = "t4" ]; then

    # TSUBAME4

    source ~/.bash_profile

    module purge
    module load cuda/12.8.0
    module load openmpi/5.0.7-gcc

    # FIXME: multiple processes on the same node
    HOSTS=$(cat $PE_HOSTFILE | awk '{print$1}' | paste -s -d ',')

    N_NODES=$(cat $PE_HOSTFILE | wc -l)
    N_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    WORLD_SIZE=$(($N_NODES * $N_PER_NODE))

    MASTER_ADDR=$(hostname -i)
    MASTER_PORT=12345

    mpirun -np $WORLD_SIZE \
        --npernode $N_PER_NODE \
        -H $HOSTS \
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

    MASTER_ADDR="localhost"
    MASTER_PORT=12345

    apptainer \
        exec \
        --nv \
        --env "MASTER_ADDR=$MASTER_ADDR" \
        --env "MASTER_PORT=$MASTER_PORT" \
        --bind .:/workspace \
        apptainer/pytorch.sif \
        ./pretrain_gpt/scripts/train-fp8.sh
fi
