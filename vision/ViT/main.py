import math
import argparse

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
from torchvision.models import vision_transformer

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from te_patch import patch_linear_norm

from trainer_fp32 import train, test
from trainer_amp import train_amp

class cfg:
    epoch = 10
    lr = 5e-3
    batch_size = 4096
    mini_batch_size = 128
    accum_step = batch_size // mini_batch_size


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argument():
    parser = argparse.ArgumentParser('Pre-training VisionTransformer with ImageNet')
    parser.add_argument('--model', default='B16', choices=['B16', 'B32', 'L16', 'L32', 'H14'])
    parser.add_argument('--compute-dtype', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--weight-dtype', default='fp32', choices=['fp32', 'fp16', 'fp8'])
    parser.add_argument('--tensorboard-logdir')
    parser.add_argument('--use-random-input', action='store_false')
    return parser.parse_args()


_to_torch_dype = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


if __name__ == '__main__':
    
    args = argument()
    
        
    if args.tensorboard_logdir is None:
        writer = None
    else:
        writer = SummaryWriter(args.tensorboard_logdir)
        
    
    
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    train_ds = datasets.ImageFolder('vision/data/ILSVRC2012_img_train', transform=train_trans)
    valid_ds = datasets.ImageFolder('vision/data/ILSVRC2012_img_val', transform=val_trans)
    
    loader_kwargs = {
        "batch_size": cfg.mini_batch_size,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, **loader_kwargs)
    
    # model = vision_transformer.vit_b_16()
    model = vision_transformer.vit_h_14()
    model = model.to(_to_torch_dype[args.weight_dtype])
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
       optimizer=optimizer,
       max_lr=cfg.lr,
       epochs=cfg.epoch,
       steps_per_epoch=math.ceil(len(train_loader) / cfg.accum_step),
   )
    
    for epoch in range(1, cfg.epoch+1):
        
        print('-'*5, f"[Epoch{epoch:02}]", '-'*5)
        
        if args.compute_dtype == 'fp32':
            train(model, train_loader, optimizer, scheduler, epoch,  
                  grad_accum_step=cfg.accum_step, tensorboard_writer=writer)
            
        elif args.compute_dtype in ['fp16', 'bf16']:
            # bfloat16 does not need scale 
            # because the exponential part is the same as IEEE754
            scaler = torch.amp.GradScaler(enabled=(args.compute_dtype == 'fp16'))
            dtype  = _to_torch_dype[args.compute_dtype]
            train_amp(model, train_loader, optimizer, scheduler, epoch, 
                      scaler, dtype, 
                      grad_accum_step=cfg.accum_step, tensorboard_writer=writer)
            
        test(model, valid_loader, epoch, tensorboard_writer=writer)
        
    