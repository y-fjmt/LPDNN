import torch
from torchvision import transforms
from typing import Literal

torch_dype_dict = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

def to_torch_dype(str_dtype: Literal['fp32', 'fp16', 'bf16']):
    return torch_dype_dict[str_dtype]


def get_transforms() -> tuple[transforms.Compose]:

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
    
    return (train_trans, val_trans)