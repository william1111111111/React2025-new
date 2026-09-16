import hashlib,io
from pathlib import Path
import torch
from reward_policy.common import read,write,sha,PARENT
OUT=Path('runs/reaction_flow/generator_reward_nft_v1')
OLD=Path('runs/reaction_flow/reward_policy_v1')
V2=Path('runs/reaction_flow/reward_signal_v2')
ARMS=('N0-quality','N1-quality-coverage')

def digest(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():h.update(name.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
