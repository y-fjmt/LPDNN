import json
import argparse
from argparse import Namespace

import torch
from transformer_engine.common.recipe import Format

to_torch_dype = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

to_te_fp8_format = {
    'hybrid': Format.HYBRID,
    'e4m3': Format.E4M3,
    'e5m2': Format.E5M2,
}


def argument(summary: bool = True) -> Namespace:
    
    parser = argparse.ArgumentParser('Pre-training VisionTransformer with ImageNet')
    
    # model settings
    parser.add_argument('--model', default='b16', 
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'],
                        help='Vision Transformer model variant',
                        dest='model')
    
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Model dropout late',
                        dest='dropout')
    
    parser.add_argument('--dtype', default='fp32', 
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Computational data type',
                        dest='dtype')
    
    parser.add_argument('--weight-dtype', default='fp32', 
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Model weight data type',
                        dest='weight_dtype')
    
    parser.add_argument('--fp8', action='store_true',
                        help='Flag to enable fp8 training',
                        dest='fp8')
    
    parser.add_argument('--fp8-format', default='hybrid', 
                        choices=['hybrid', 'e4m3', 'e5m2'],
                        help='fp8 format to compute specific layer',
                        dest='fp8_format')
    
    # training setting
    parser.add_argument('--lr', type=float,
                        help='Max learning rate in the LR scheduler',
                        dest='lr')
    
    parser.add_argument('--epoch', type=int,
                        help='Number of iteration',
                        dest='epoch')
    
    parser.add_argument('--batch-size', type=int,
                        help='Number of data used for parameter update',
                        dest='batch_size')
    
    parser.add_argument('--mini-batch-size', type=int,
                        help='Number of data for fw/bw computation',
                        dest='mini_batch_size')
    
    
    # dataset setting
    parser.add_argument('--imagenet-root',
                        help='Path to imagenet dataset',
                        dest='imagenet_root')
    
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers for dataloader',
                        dest='workers')
    
    
    # misc
    parser.add_argument('--tensorboard-logdir', default=None,
                        help='Path to tensorboard log directory')
    
    parser.add_argument('--debug', action='store_true',
                        help='Flag to enable debug mode',
                        dest='debug')
    
    
    args = parser.parse_args()
    args.accum_step = args.batch_size // args.mini_batch_size
    
    if summary:
        args_dict = vars(args)
        print('='*20, 'Arguments', '='*20)
        print(json.dumps(args_dict, indent=2))
        print('='*50)
        
    # str -> actual type
    args.dtype = to_torch_dype[args.dtype]
    args.weight_dtype = to_torch_dype[args.weight_dtype]
    args.fp8_format = to_te_fp8_format[args.fp8_format]
    
    return args