import json
from pathlib import Path
import numpy as np
import torch
from hirp.train_phase25 import TrainConfig
from mam_target.data import TaskData
from hirp.phase15_audit import sha256_file
ROOT=Path('runs/mam_target/staged_v2');SCHEDULE=Path('runs/mam_target/refine_v1')
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def resources(device):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert sha256_file(manifest['data_manifest'])==manifest['data_manifest_sha256']
    assert sha256_file(manifest['parent_checkpoint'])==manifest['parent_sha256']
    assert sha256_file(SCHEDULE/'schedule.json')==manifest['schedule_sha256'] and sha256_file(SCHEDULE/'noise.npy')==manifest['noise_sha256']
    records=json.loads((SCHEDULE/'schedule.json').read_text())['records'];noise=torch.from_numpy(np.load(SCHEDULE/'noise.npy'))
    data=TaskData(json.loads(Path(manifest['data_manifest']).read_text()),TrainConfig(max_steps=8000),'C0',device)
    assert len(records)==len(noise)==2000
    return manifest,records,noise,data
