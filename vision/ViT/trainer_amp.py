import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import amp
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_amp(model: nn.Module, train_loader: DataLoader, 
              optimizer: optim.Optimizer, device: torch.device,
              scaler: amp.GradScaler, dtype: torch.dtype) -> None:
    
    model.train()
    model = model.to(device)
    
    iter = tqdm(train_loader)
    iter.set_description('training progress')
    
    for inputs, labels in iter:
        
        optimizer.zero_grad()
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda', dtype):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # scaler.unscale_(optimizer)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        iter.set_postfix({'loss': loss.detach().item()})
        
