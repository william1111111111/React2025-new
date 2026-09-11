from dataclasses import asdict
import torch
from hirp.phase22 import HiRP22
from hirp.phase21 import OutputConfig, Phase21OutputHead
from hirp.config import HiRPConfig

class UnboundedResidualHead(Phase21OutputHead):
    def forward(self,h,decoded,valid_mask):
        raw=self.raw_probe(self.base(self.base_norm(h))[:,None]+self.stochastic(self.stochastic_norm(decoded)))
        y=torch.cat((raw[...,:15].sigmoid(),raw[...,15:17].tanh(),raw[...,17:].softmax(-1)),-1)
        return y.masked_fill(~valid_mask[:,None,:,None],0)

class TaskHiRP(HiRP22):
    def __init__(self,config):
        output=OutputConfig(head_pre_norm=True,projection_init='small',small_init_std=.01)
        super().__init__(config,output,'standard_normal')
        # Same parameters, but explicitly versioned changed function. No legacy loading.
        self.output_head.__class__=UnboundedResidualHead

def make_model(seed=123,device='cpu',config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return TaskHiRP(config or HiRPConfig()).to(device)

def metadata(model):
    return dict(format_version='mam-target-unbounded-v1',config=asdict(model.config),output_config=asdict(model.output_config),prior_mode=model.prior_mode,initialization='scratch; small nonzero projections std .01; original nonzero FiLM')

def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format_version')!='mam-target-unbounded-v1':raise ValueError('not a task variant checkpoint')
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):model=TaskHiRP(HiRPConfig(**saved['config']))
    if metadata(model)!= {k:saved[k] for k in metadata(model)}:raise ValueError('model metadata mismatch')
    model.load_state_dict(saved['model']);return model.to(device).eval(),saved
