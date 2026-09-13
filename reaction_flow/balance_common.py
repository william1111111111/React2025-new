"""Frozen inputs and initialization for the two-arm balance experiment."""
import json
from pathlib import Path
import torch
from .config import FlowConfig, ROOT as AROOT, DATA_MANIFEST
from .data import FlowData
from .shared_model import load_checkpoint
from .balance_controller import RatioController
from hirp.phase15_audit import sha256_file, canonical_hash
from hirp.train_phase25 import restore_rng
from mam_refine.train import tree_hash
ROOT=Path('runs/reaction_flow/prior_task_balance_v1')
PARENT=Path('runs/reaction_flow/shared_noise_v1/G1-shared/attempt_000/checkpoints/step_016000.pt')
PARENT_SHA='5a8dc94daf43530b0f304be38e74d27e384c4eac23b1d346e48ff3179c31e792'
ARMS=('B0-fixed','B1-ratio')

def resources(device='cuda:0'):
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    for p,h in protocol['inputs'].items():
        if sha256_file(p)!=h:raise ValueError('changed input '+p)
    records=json.loads((ROOT/'schedule.json').read_text())['records']
    assert len(records)==2000 and [r['step'] for r in records]==list(range(16001,18001))
    return FlowConfig(),records,FlowData(FlowConfig(),device),protocol

def initialize(arm,device='cuda:0',resume=None):
    assert arm in ARMS
    assert sha256_file(PARENT)==PARENT_SHA
    model,saved=load_checkpoint(resume or PARENT,device);model.train()
    assert model.rho==.05
    params=[p for p in model.velocity.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);opt.load_state_dict(saved['optimizer'])
    for g in opt.param_groups:g['lr']=2e-5
    if resume:
        assert saved['phase']==arm and saved['parent_sha256']==PARENT_SHA
        assert saved['adaptation_protocol_sha256']==sha256_file(ROOT/'PROTOCOL.json')
        assert saved['calibration_sha256']==sha256_file(ROOT/'calibration.json')
        control=RatioController.restore(saved['ratio_controller'],arm)
        assert saved['phase_step']==control.completed==len(saved['rows'])
    else:
        assert saved['phase']=='G1-shared' and saved['global_step']==16000 and saved['completed']
        cal=json.loads((ROOT/'calibration.json').read_text());assert cal['parent_sha256']==PARENT_SHA
        control=RatioController(cal['A'],cal['B'])
    restore_rng(saved['rng'])
    return model,opt,params,control,saved

def gradients(fm,task,params):
    def vector(loss):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        if not any(g is not None for g in gs):raise ValueError('all gradients unused')
        v=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
        if not torch.isfinite(v).all():raise ValueError('nonfinite component gradient')
        return v
    gf,gq=vector(fm),vector(task);a,b=float(gf.norm()),float(gq.norm())
    RatioController.validate_norms(a,b)
    return a,b,float(torch.nn.functional.cosine_similarity(gf,gq,dim=0))
