from io import BytesIO

from torch import nn
import transformer_engine.pytorch as te

def remove_extra_state_from_state_dict(self, destination, prefix, local_metadata):
   """
   HFのsave_pretrained()メソッドがBytesIO型を保存できないため、保存前に削除するhook
   """
   for key in list(destination.keys()):
       if key.endswith('._extra_state') and isinstance(destination[key], BytesIO):
           del destination[key]


def patch_linear_norm(model):
   for name, module in model.named_children():
       if isinstance(module, nn.Linear):
           # Tensor Coreの制約のため次元が16の倍数である必要がある
           if any(p % 16 != 0 for p in module.weight.shape):
               return
           has_bias = module.bias is not None
           te_module = te.Linear(
               module.in_features, module.out_features, bias=has_bias,
               params_dtype=module.weight.dtype
           )
           te_module.weight.copy_(module.weight)
           if has_bias:
               te_module.bias.copy_(module.bias)
           te_module._register_state_dict_hook(remove_extra_state_from_state_dict)

           setattr(model, name, te_module)

       elif isinstance(module, nn.LayerNorm):
           te_module = te.LayerNorm(
               module.normalized_shape[0], eps=module.eps,
               params_dtype=module.weight.dtype
           )
           te_module.weight.copy_(module.weight)
           te_module.bias.copy_(module.bias)
           te_module._register_state_dict_hook(remove_extra_state_from_state_dict)

           setattr(model, name, te_module)

       else:
           patch_linear_norm(module)