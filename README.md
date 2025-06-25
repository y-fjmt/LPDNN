# Low-Precision Deep Neural Networks

## Overview
This repository provides experimental setups for evaluating the performance of DNNs in terms of accuracy and computational efficiency.
Also, This repository capable with [TSUBAME4](https://www.t4.cii.isct.ac.jp/).

## Setup
LPDNN depends on [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).  
It has already been added as a Git submodule, so please clone this repository with
the  `--recursive` option to include Megatron-LM as well.
These recipes are intended to run on the Apptainer container system.
You can build the container using the following command. It takes tens of minutes.  
```bash
./apptainer/setup.sh
```  

> [!NOTE]
> In TSUBAME4, commands are basically executed with `qsub`.
>
>     qsub -g tga-hoge ./apptainer/setup.sh

## Pretrain GPT3-345M with C4 dataset
Download and Preprocess dataset.
```
./pretrain_gpt/prepare.sh
```
Training (FP8)
```
./pretrain_gpt/train-fp8.sh
```

