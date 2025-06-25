#!/bin/bash

apptainer \
    shell \
    --nv \
    -B .:/workspace \
    apptainer/pytorch.sif