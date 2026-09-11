"""Frozen TRAIN source/target streams; uniform slots independent of model and noise."""
import hashlib,json
from pathlib import Path
import torch
from mam_target.data import TaskData
from hirp.train_phase25 import TrainConfig
from hirp.phase15_audit import sha256_file
from .config import ROOT,DATA_MANIFEST

def stream_seed(seed,step,tag):
    return int.from_bytes(hashlib.sha256(f'{seed}|{step}|{tag}'.encode()).digest()[:8],'big')%(2**63-1)

def draw(shape,seed,step,tag,normal=True):
    rng=torch.Generator().manual_seed(stream_seed(seed,step,tag))
    return torch.randn(shape,generator=rng) if normal else torch.rand(shape,generator=rng)

class FlowData:
    def __init__(self,cfg,device='cpu'):
        self.config=cfg;self.device=device
        self.inner=TaskData(json.loads(DATA_MANIFEST.read_text()),TrainConfig(max_steps=14000,T=cfg.T,sampler_seed=cfg.sampler_seed),'C0',device)
    def batch(self,record):
        batch,_,targets,lengths,ids=self.inner.task_batch(record)
        slots=torch.tensor(record['target_slots'],device=self.device)
        index=torch.arange(len(slots),device=self.device);selected=targets[index,slots];selected_lengths=lengths[index,slots]
        return batch,targets,lengths,ids,selected,selected_lengths,slots

def resources(cfg,device='cpu'):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name in ['schedule.json','coordinate_stats.json']:
        if sha256_file(ROOT/name)!=manifest['artifact_hashes'][name]:raise ValueError('changed '+name)
    if sha256_file(DATA_MANIFEST)!=manifest['data_manifest_sha256']:raise ValueError('changed training provenance')
    records=json.loads((ROOT/'schedule.json').read_text())['records']
    if len(records)!=cfg.steps+cfg.adaptation_steps:raise ValueError('incomplete schedule')
    return manifest,records,FlowData(cfg,device)
