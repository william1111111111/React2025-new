import json
from pathlib import Path
import numpy as np
import torch
from hirp.train_phase25 import TrainConfig,RealData,train
from hirp.train_phase21 import configure
configure()
root=Path('runs/phase25/timescale_v1');m=json.load(open(root/'seed_123/manifest.json'));cfg=TrainConfig()
records=json.load(open(root/'seed_123/schedule.json'))['records'];noise=torch.from_numpy(np.load(root/'seed_123/noise.npy'))
for arm in ('C0','C1','C2'):
    data=RealData(m,cfg,arm,'cuda:0')
    result,model,opt=train(root/'gpu0_smoke'/arm,cfg,arm,records,noise,data,m,'cuda:0',stop_after=3)
    print('RESULT',arm,json.dumps(result),flush=True)
    del data,model,opt
    torch.cuda.empty_cache()
