import torch
from torch import nn
from typing import Literal
from torchvision.models import vision_transformer

def vit_initializer(
        key: Literal['b16', 'b32', 'l16', 'l32', 'h14'], 
        dtype: torch.dtype = torch.float32,
    ) -> nn.Module:
    
    if key == 'b16':
        model = vision_transformer.vit_b_16()
    elif key == 'b32':
        model = vision_transformer.vit_b_32()
    elif key == 'l16':
        model = vision_transformer.vit_l_16()
    elif key == 'l32':
        model = vision_transformer.vit_l_32()
    elif key == 'h14':
        model = vision_transformer.vit_h_14()
    else:
        raise ValueError(f'ViT-{key} is not defined in torchvision')
    
    return model.to(dtype)