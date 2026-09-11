"""Real T750 data: finite updates, frozen parameters and exact 2 vs 1+1 resume."""
import torch
from mam_staged.train import train
from mam_staged.common import ROOT,write
from mam_refine.train import tree_hash
from hirp.train_phase21 import configure

def main():
    configure();torch.use_deterministic_algorithms(True);paths={}
    for arm in ('S0_quality','S1_joint','S2_staged'):
        paths[arm]=train(arm,2,'smoke_'+arm)
    split=train('S2_staged',1,'smoke_split')
    resumed=train('S2_staged',2,'smoke_resume',split/'checkpoints/step_006001.pt')
    a=torch.load(paths['S2_staged']/'checkpoints/step_006002.pt',map_location='cpu',weights_only=True)
    b=torch.load(resumed/'checkpoints/step_006002.pt',map_location='cpu',weights_only=True)
    error=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    assert error==0 and tree_hash(a['optimizer'])==tree_hash(b['optimizer']) and a['new_rows']==b['new_rows']
    assert tree_hash(a['rng'])==tree_hash(b['rng'])
    write(ROOT/'gpu_regression.json',dict(resume_parameter_max_error=error,optimizer_equal=True,rng_equal=True,rows_equal=True,actual_steps=2,continuous_vs_split='2 vs 1+1',smoke_paths={k:str(v) for k,v in paths.items()},resume_path=str(resumed),full_T=750,B=4,K=4))
if __name__=='__main__':main()
