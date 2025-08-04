import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import amp
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_amp(
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        device: torch.device,
        scaler: amp.GradScaler, 
        dtype: torch.dtype,
        grad_accum_step: int = 1
    ) -> None:
    
    model.train()
    model = model.to(device)
    
    optimizer.zero_grad()
    
    iter = tqdm(train_loader)
    iter.set_description('training progress')
    
    for batch_idx, (inputs, labels) in enumerate(iter):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda', dtype):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss = loss / grad_accum_step
        
        scaler.scale(loss).backward()
        
        if ((batch_idx+1) % grad_accum_step == 0) or (batch_idx+1 == len(iter)):
            # scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        iter.set_postfix({'loss': loss.detach().item() * grad_accum_step})
        
