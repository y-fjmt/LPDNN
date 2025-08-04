import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train(
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        device: torch.device,
        grad_accum_step: int = 1
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
        
        if ((batch_idx+1) % grad_accum_step == 0) or (batch_idx+1 == len(iter)):
            optimizer.step()
            optimizer.zero_grad()
        
        iter.set_postfix({'loss': loss.detach().item() * grad_accum_step})
        


def test(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: torch.device
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
