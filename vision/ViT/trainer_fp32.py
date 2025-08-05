import math
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler.LRScheduler,
        epoch: int,
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
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss = loss / grad_accum_step
        loss.backward()
        
        scaled_loss = loss.detach().item() * grad_accum_step
        
        if ((batch_idx+1) % grad_accum_step == 0) or (batch_idx+1 == len(iter)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            if tensorboard_writer is not None:
                steps_per_epoch = math.ceil(len(train_loader) / grad_accum_step)
                global_step = (epoch-1) * steps_per_epoch + batch_idx // grad_accum_step
                tensorboard_writer.add_scalar('train_loss', scaled_loss, global_step)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
        
        iter.set_postfix({'loss': scaled_loss})
        


def test(
        model: nn.Module, 
        test_loader: DataLoader, 
        epoch: int,
        tensorboard_writer: SummaryWriter | None = None,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
    
    model.eval()
    model = model.to(device)
    
    test_loss = 0.0
    n_correct = 0
    
    with torch.no_grad():
        
        iter = tqdm(test_loader)
        iter.set_description('validation progress')
        
        for inputs, labels in iter:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            
            pred = outputs.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(labels.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {n_correct}/{len(test_loader.dataset)} "
        f"({100. * n_correct / len(test_loader.dataset):.0f}%)\n"
    )
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('valid_loss', test_loss, epoch)
        tensorboard_writer.add_scalar('accuracy', 100. * n_correct / len(test_loader.dataset), epoch)
