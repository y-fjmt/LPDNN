import math
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import amp
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_amp(
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler.LRScheduler,
        epoch: int,
        scaler: amp.GradScaler, 
        dtype: torch.dtype,
        grad_accum_step: int = 1,
        tensorboard_writer: SummaryWriter | None = None,
        device: torch.device = DEFAULT_DEVICE,
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
        
        scaled_loss = loss.detach().item() * grad_accum_step
        
        if ((batch_idx+1) % grad_accum_step == 0) or (batch_idx+1 == len(iter)):
            # scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            if tensorboard_writer is not None:
                steps_per_epoch = math.ceil(len(train_loader) / grad_accum_step)
                global_step = (epoch-1) * steps_per_epoch + batch_idx // grad_accum_step
                tensorboard_writer.add_scalar('train_loss', scaled_loss, global_step)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            
        iter.set_postfix({'loss': scaled_loss})
        
