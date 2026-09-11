import torch
BASE_LR={'source_base':1e-5,'decoder_body':2e-5,'stochastic':5e-5}

def partition(model):
    groups={k:[] for k in (*BASE_LR,'prior')}
    for name,p in model.named_parameters():
        if name.startswith('prior.'):key='prior'
        elif '.film1.projection.' in name or '.film2.projection.' in name or name.startswith(('output_head.stochastic.','output_head.stochastic_norm.')):key='stochastic'
        elif name.startswith('decoder.'):key='decoder_body'
        elif name.startswith(('stems.','encoder.','output_head.base.','output_head.base_norm.')):key='source_base'
        else:raise ValueError('unclassified '+name)
        groups[key].append((name,p))
    assert sum(len(v) for v in groups.values())==len(list(model.parameters()))
    assert len({id(p) for v in groups.values() for _,p in v})==len(list(model.parameters()))
    return groups

def make_optimizer(model):
    groups=partition(model)
    return torch.optim.AdamW([dict(params=[p for _,p in groups[k]],lr=lr,name=k) for k,lr in BASE_LR.items()],weight_decay=.01)

def apply_stage(model,optimizer,arm,u):
    # u is zero-based NEW update index. First update uses u=0.
    groups=partition(model);phase='warmup' if u<500 else 'decoder' if u<1500 else 'final_half_lr'
    for name,items in groups.items():
        active=name!='prior' and (arm!='S2_staged' or name=='stochastic' or (u>=500 and name=='decoder_body'))
        for _,p in items:
            p.requires_grad_(active)
            if not active:p.grad=None
    for group in optimizer.param_groups:group['lr']=BASE_LR[group['name']]*(.5 if u>=1500 else 1.)
    model.eval()
    return phase
