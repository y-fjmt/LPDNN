#!/bin/bash
#$-cwd
#$-l cpu_4=1
#$-l h_rt=24:00:00

pwd=$(pwd)

# ImageNet 1K
dist="vision/data/imagenet"
mkdir -p $dist
cd $dist

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

tar -zxvf ILSVRC2012_img_train.tar
tar -zxvf ILSVRC2012_img_val.tar
tar -zxvf ILSVRC2012_devkit_t12.tar.gz

apptainer \
    exec \
    --nv \
    --bind .:/workspace \
    apptainer/pytorch.sif \
    python3 extract_imagenet.py --train --valid

cd $pwd