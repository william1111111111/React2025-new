"""Differentiable output-path probes and component-gradient audits (no losses added)."""
from contextlib import contextmanager
import math
import torch
from .phase21 import forward_a0, conditional_score, CHANNEL_SCALE
from .paired_data import paired_model_inputs
from .phase1_diagnostics import gradient_norm

GROUPS = {'AU': slice(0, 15), 'VA': slice(15, 17), 'expression': slice(17, 25)}


def stats(x):
    x = x.detach().float().flatten()
    q = torch.quantile(x.abs(), x.new_tensor([0., .05, .5, .95, .99, 1.])).tolist()
    return dict(mean=x.mean().item(), std=x.std(unbiased=False).item(),
                abs_quantiles=dict(zip(('min','p05','p50','p95','p99','max'), q)))


def module_norms(model, gradients, params):
    lookup = {id(p): g for p, g in zip(params, gradients)}
    result = {}
    for name, module in [('total',model), ('stems',model.stems), ('encoder',model.encoder),
                         ('decoder',model.decoder), ('base_projection',model.output_head.base),
                         ('stochastic_projection',model.output_head.stochastic), ('prior',model.prior)]:
        gs = [lookup[id(p)] for p in module.parameters() if lookup.get(id(p)) is not None]
        result[name] = math.sqrt(sum(g.detach().float().square().sum().item() for g in gs))
    return result


def component_gradients(model, conditional, group, weight):
    params = [p for p in model.parameters() if p.requires_grad]
    a = torch.autograd.grad(conditional, params, retain_graph=True, allow_unused=True)
    b = (torch.autograd.grad(group, params, retain_graph=True, allow_unused=True)
         if group is not None else [None] * len(params))
    an, bn = module_norms(model,a,params), module_norms(model,b,params)
    dot = sum((x.detach().float()*y.detach().float()).sum().item()
              for x,y in zip(a,b) if x is not None and y is not None)
    return dict(conditional=an, group=bn, weighted_group_norm=weight*bn['total'],
                weighted_group_over_conditional=weight*bn['total']/an['total'] if an['total'] else None,
                cosine=dot/(an['total']*bn['total']) if an['total']*bn['total'] else None,
                finite=all(torch.isfinite(g).all().item() for g in list(a)+list(b) if g is not None))


@contextmanager
def capture_path(model):
    captured = {}; hooks = []
    def save(name, first=False):
        def hook(module, inputs, output):
            captured[name] = output[0] if first else output
        return hook
    for name, module, first in [('encoder_H',model.encoder,True), ('decoder_hidden',model.decoder,False),
                                ('base_logits',model.output_head.base,False),
                                ('stochastic_logits',model.output_head.stochastic,False),
                                ('final_raw',model.output_head.raw_probe,False)]:
        hooks.append(module.register_forward_hook(save(name,first)))
    try:
        yield captured
    finally:
        for hook in hooks: hook.remove()


def output_path_diagnostic(model, batch, noise, channel_scale=CHANNEL_SCALE):
    """Fixed eval probe; gradients of conditional ES at intermediates and dY/dnoise.

    Noise metric is ||J^T v|| for a seeded Rademacher output cotangent v,
    normalized by sqrt(number of valid scalar outputs), per output group.
    This probes output Jacobians independently of loss-gradient cancellation.
    Does not alter parameter .grad, RNG streams, or module train/eval modes.
    """
    modes = [(m,m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.enable_grad(), capture_path(model) as captured:
            epsilon = noise.detach().clone().requires_grad_(True)
            pred = forward_a0(model, **paired_model_inputs(batch), sample_count=noise.shape[1], noise=epsilon)
            score = conditional_score(pred, batch, channel_scale)
            tensors = list(captured.values())
            grads = torch.autograd.grad(score['loss'], tensors, retain_graph=True, allow_unused=True)
            valid = torch.arange(pred.shape[2],device=pred.device)[None] < batch['source_lengths'][:,None]
            result = dict(conditional={k:v.item() for k,v in score.items()}, paths={}, outputs={}, noise_jacobian={})
            for (name,value), grad in zip(captured.items(),grads):
                mask = valid if value.ndim == 3 else valid[:,None].expand(value.shape[:3])
                groups = GROUPS if value.shape[-1] == 25 else {'all':slice(None)}
                result['paths'][name] = {}
                for group, sl in groups.items():
                    x = value[...,sl][mask]
                    result['paths'][name][group] = dict(**stats(x),
                        conditional_gradient_norm=grad[...,sl][mask].float().norm().item() if grad is not None else 0.)
            mask = valid[:,None].expand(pred.shape[:3])
            raw = captured['stochastic_logits'][mask].float()
            result['residual'] = dict(tanh_saturation_fraction=(raw.tanh().abs()>=.99).float().mean().item(),
                derivative=stats(1-raw.tanh().square()), abs_mean=raw.tanh().abs().mean().item())
            for group, sl in GROUPS.items():
                q = pred[...,sl]; values = q[mask]
                low = -1 if group == 'VA' else 0
                result['outputs'][group] = dict(**stats(values),
                    boundary_fraction=((values<=low+.01)|(values>=.99)).float().mean().item(),
                    prediction_std=q.float().std(1,unbiased=False)[valid].mean().item())
                rng = torch.Generator(device=pred.device).manual_seed(2105)
                v = (torch.randint(2,q.shape,generator=rng,device=q.device)*2-1).to(q.dtype)
                v = v.masked_fill(~mask[...,None],0)/math.sqrt(values.numel())
                g = torch.autograd.grad((q*v).sum(),epsilon,retain_graph=True)[0]
                result['noise_jacobian'][group] = dict(vjp_norm=g.float().norm().item(),finite=torch.isfinite(g).all().item())
            return result
    finally:
        for module,mode in modes: module.training = mode
