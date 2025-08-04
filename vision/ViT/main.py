import argparse

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
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
    mini_batch_size = 64
    accum_step = batch_size // mini_batch_size


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RandomDataset(Dataset):
    
    def __init__(self, n_data: int, shape: torch.Size, 
                 dtype: torch.dtype = torch.float32, 
                 device: torch.device = torch.device('cpu')
        ) -> None:
        super().__init__()
        self.n_data = n_data
        self.shape = shape
        self.dtype = dtype
        self.device = device
        
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, index: int) -> torch.Tensor:
        input = torch.rand(self.shape, dtype=self.dtype, device=self.device)
        label = torch.randint(0, 1000, (1,), device=self.device).item()
        return (input, label)

def argument():
    parser = argparse.ArgumentParser('Pre-training VisionTransformer with ImageNet')
    parser.add_argument('--model', default='B16', choices=['B16', 'B32', 'L16', 'L32', 'H14'])
    parser.add_argument('--compute-dtype', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--weight-dtype', default='fp32', choices=['fp32', 'fp16', 'fp8'])
    parser.add_argument('--use-random-input', action='store_false')
    return parser.parse_args()


_to_torch_dype = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


if __name__ == '__main__':
    
    args = argument()
    
    # model = vision_transformer.vit_b_16()
    model = vision_transformer.vit_h_14()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    model = model.to(_to_torch_dype[args.weight_dtype])
    
    if args.compute_dtype in ['fp16', 'bf16']:
        scaler = torch.amp.GradScaler()
    
    # ds_kwargs = {
    #     "shape": (3, 224, 224),
    #     "dtype": torch.float32,
    #     "device": 'cpu',
    # }
    # train_ds = RandomDataset(65536, **ds_kwargs)
    # valid_ds = RandomDataset(4096, **ds_kwargs)
    
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
    valid_ds = datasets.ImageFolder('vision/data/ILSVRC2012_img_train', transform=train_trans)
    
    loader_kwargs = {
        "batch_size": cfg.mini_batch_size,
        "shuffle": True,
        "num_workers": 128,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, **loader_kwargs)
    
    for epoch in range(1, cfg.epoch+1):
        
        print('-'*5, f"[Epoch{epoch:02}]", '-'*5)
        
        if args.compute_dtype == 'fp32':
            train(model, train_loader, optimizer, device, grad_accum_step=cfg.accum_step)
        elif args.compute_dtype == 'fp16':
            train_amp(model, train_loader, optimizer, device, scaler, torch.float16, grad_accum_step=cfg.accum_step)
        elif args.compute_dtype == 'bf16':
            train_amp(model, train_loader, optimizer, device, scaler, torch.bfloat16, grad_accum_step=cfg.accum_step)
            
        test(model, valid_loader, device)
        
    