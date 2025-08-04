#!/bin/bash
#$-cwd
#$-l cpu_4=1
#$-l h_rt=00:05:00

pwd=$(pwd)

# MNIST
dist="vision/data/MNIST/raw"
mkdir -p $dist
cd $dist

wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz

cd $pwd

# ImageNet 1K
dist="imagenet"
mkdir -p $dist
cd $dist

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

tar -zxvf ILSVRC2012_img_train.tar
tar -zxvf ILSVRC2012_img_val.tar
tar -zxvf ILSVRC2012_devkit_t12.tar.gz
python3 extract_imagenet.py --train --valid

cd $pwd