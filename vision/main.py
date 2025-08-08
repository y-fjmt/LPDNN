import math

import torch
from torch import optim
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets

from transformer_engine.common.recipe import DelayedScaling

from argument import argument
from te_patch import patch_linear_norm
from model import vit_initializer
from trainer_fp32 import train, test
from trainer_amp import train_amp
from utils import get_transforms


if __name__ == '__main__':
    
    args = argument()
    
    if args.tensorboard_logdir is None:
        writer = None
    else:
        writer = SummaryWriter(args.tensorboard_logdir)
    
    
    # prepare dataset
    (train_trans, val_trans) = get_transforms()
    train_ds = datasets.ImageFolder(args.imagenet_root + '/ILSVRC2012_img_train', 
                                    transform=train_trans)
    valid_ds = datasets.ImageFolder(args.imagenet_root + '/ILSVRC2012_img_val_classified', 
                                    transform=val_trans)
    
    if args.debug:
        indices = range(2 * args.batch_size)
        train_ds = Subset(train_ds, indices)
        valid_ds = Subset(valid_ds, indices)
    
    train_loader = DataLoader(train_ds, args.mini_batch_size, True, 
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, args.mini_batch_size, True, 
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    
    
    # model and training componets
    model = vit_initializer(args.model, dropout=args.dropout, dtype=args.weight_dtype)
    
    # replace LayerNorm and Linear to TE implementation
    fp8_recipe = None
    if args.fp8:
        fp8_recipe = DelayedScaling(fp8_format=args.fp8_format)
        with torch.no_grad():
            patch_linear_norm(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
       optimizer=optimizer,
       max_lr=args.lr,
       epochs=args.epoch,
       steps_per_epoch=math.ceil(len(train_loader) / args.accum_step),
       pct_start=0.1
   )
    
    
    # trainint and validation
    for epoch in range(1, args.epoch+1):
        
        print('-'*5, f"[Epoch{epoch:02}]", '-'*5)
        
        if args.dtype is torch.float32:
            train(model, train_loader, optimizer, scheduler, epoch,  
                  grad_accum_step=args.accum_step, tensorboard_writer=writer)
            
        elif (args.dtype is torch.float16) or (args.dtype is torch.bfloat16):
            # bfloat16 does not need scale 
            # because the exponential part is the same as IEEE754
            scaler = torch.amp.GradScaler(enabled=(args.dtype is torch.float16))
            train_amp(model, train_loader, optimizer, scheduler, epoch, 
                      scaler, args.dtype, fp8_recipe=fp8_recipe,
                      grad_accum_step=args.accum_step, tensorboard_writer=writer)
            
        test(model, valid_loader, epoch, tensorboard_writer=writer)
        
    