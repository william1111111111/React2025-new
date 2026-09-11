from dataclasses import dataclass,asdict
from pathlib import Path
ROOT=Path('runs/reaction_flow/trajectory_v1')
PARENT=Path('runs/mam_target/task_v1/R2/attempt_001/checkpoints/step_006000.pt')
PARENT_SHA='d9b0e07aad91f3bbb254bfc2ad5780f9fa2b9a138245dbafd0c79fa935bf8936'
DATA_MANIFEST=Path('runs/phase25/timescale_v1/seed_123/manifest.json')
@dataclass(frozen=True)
class FlowConfig:
    d_model:int=256
    blocks:int=8
    heads:int=8
    ff_width:int=1024
    dropout:float=0.
    coordinates:int=24
    eps:float=1e-4
    std_floor:float=1e-3
    projection_std:float=.01
    integration_steps:int=16
    seed:int=123
    sampler_seed:int=123
    target_seed:int=3000126
    noise_seed:int=4000126
    tau_seed:int=5000126
    rollout_seed:int=6000126
    evaluation_seed:int=7100123
    T:int=750
    B:int=4
    steps:int=12000
    adaptation_steps:int=2000
    lr:float=1e-4
    weight_decay:float=.01
    statistics_batches:int=1024
    checkpoints:tuple=(2000,6000,12000)
    def dictionary(self):return asdict(self)
