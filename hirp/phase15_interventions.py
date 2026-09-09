"""Read-only prior overrides. Never modify decoder, weights, clamps or scales."""
from contextlib import contextmanager
from collections import defaultdict
import torch


def within_session_permutation(session_ids, seed=1502):
    groups = defaultdict(list)
    for i, session in enumerate(session_ids):
        groups[session].append(i)
    result = torch.arange(len(session_ids))
    rng = torch.Generator().manual_seed(seed)
    for session in sorted(groups):
        members = groups[session]
        if len(members) < 2:
            raise ValueError('within-session shuffle requires at least two inputs per session')
        order = torch.tensor(members)[torch.randperm(len(members),generator=rng)]
        result[order] = order.roll(1)
    return result


def intervention_prior(mu, log_sigma, intervention, permutation=None):
    if intervention == 'correct':
        return mu,log_sigma
    if intervention == 'shuffled':
        if permutation is None:
            raise ValueError('shuffle permutation required')
        return mu[permutation],log_sigma[permutation]
    if intervention == 'dataset_mean':
        # Arithmetic mean sigma, not exp(mean(log_sigma)); inputs only.
        return mu.mean(0,keepdim=True).expand_as(mu), log_sigma.exp().mean(0,keepdim=True).log().expand_as(log_sigma)
    raise ValueError('unknown prior intervention')


@contextmanager
def override_prior(model, mu, log_sigma):
    if model.training:
        raise ValueError('prior interventions are eval-only')
    handle = model.prior.register_forward_hook(lambda module,inputs,output:(mu,log_sigma))
    try:
        yield
    finally:
        handle.remove()
