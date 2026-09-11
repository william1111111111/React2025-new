"""Select the same unfused attention path with or without autograd on torch 2.1.
No changed weights/activation/dropout: the key clone is an identity operation.
Only training/calibration use this execution policy; public export is unchanged.
"""
import torch

def _distinct_key(module,args):
    q,k,v,*rest=args
    return (q,k.clone(),v,*rest)

def consistent_attention(model):
    for module in model.modules():
        if isinstance(module,torch.nn.TransformerEncoderLayer):
            module.activation_relu_or_gelu=0  # disable fused dispatch, keep actual activation
        if isinstance(module,torch.nn.MultiheadAttention) and not getattr(module,'_staged_unfused',False):
            module.register_forward_pre_hook(_distinct_key)
            module._staged_unfused=True
    return model
