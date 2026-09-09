"""Real-checkpoint equivalence and short GPU resume audit; no long training."""
from pathlib import Path
from dataclasses import asdict
import argparse
import torch
from torch.utils.data import default_collate
from . import HiRPConfig
from .phase21 import make_model as old_model, forward_a0, OutputConfig
from .phase22 import make_model,load_checkpoint,eval_adapter,official_adapter,model_metadata
from .train_phase22 import write,TrainConfig,schedule,train,RealData,prepare
from .train_phase21 import configure
from .session_data import SessionPopulation
from .paired_data import paired_model_inputs
from .train_phase1 import to_device,state_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:4');a=p.parse_args();configure()
    pool=SessionPopulation(Path('data'),'train');batch=to_device(default_collate([pool.dataset[i] for i in range(4)]),a.device)
    inputs=paired_model_inputs(batch);noise=torch.randn(4,32,32,generator=torch.Generator().manual_seed(2201)).to(a.device)
    result={}
    for label in ('initial','C0','C1','C2'):
        if label=='initial':
            old=old_model(a.device,OutputConfig(head_pre_norm=True)).eval();new=make_model(123,device=a.device).eval()
        else:
            path=Path('runs/phase21/conditional_score_v0/checkpoints')/(label+'.pt')
            if not path.exists():result[label]={'executed':False,'reason':'checkpoint missing'};continue
            new,saved=load_checkpoint(path,a.device,path.parent.parent/'training_manifest.json')
            old=old_model(a.device,OutputConfig(**saved['output_config']),HiRPConfig(**saved['config'])).eval();old.load_state_dict(saved['model'])
        cases=[]
        for k in (1,4,10,32):
            eps=noise[:,:k]
            with torch.no_grad():
                historical=forward_a0(old,**inputs,sample_count=k,noise=eps)
                bare=old.sample(**inputs,sample_count=k,noise=eps)
                direct=new(**inputs,sample_count=k,noise=eps)
                sample=new.sample(**inputs,sample_count=k,noise=eps)
                adapter=eval_adapter(new,**inputs,noise=eps)
                chunk=torch.cat([eval_adapter(new,**inputs,noise=eps[:,i:i+3]) for i in range(0,k,3)],1)
            errors=dict(old_hook_vs_bare=float((historical-bare).abs().max()),new_vs_old=float((direct-historical).abs().max()),
                        forward_sample=float((direct-sample).abs().max()),adapter=float((direct-adapter).abs().max()),chunks=float((direct-chunk).abs().max()))
            assert all(v<=2e-6 for key,v in errors.items() if key!='old_hook_vs_bare')
            cases.append(dict(K=k,**errors))
        scales=getattr(new,'scale_metadata',dict(channel_scale=[1.]*25,descriptor_scaler_sha256='audit-only',descriptor_dimension=75,training_T=128))
        rt=a.run/'audit_checkpoints'/(label+'.pt');rt.parent.mkdir(parents=True,exist_ok=True)
        if rt.exists():raise FileExistsError(rt)
        torch.save(dict(**model_metadata(new,scales),model=new.state_dict()),rt)
        loaded,_=load_checkpoint(rt,a.device)
        error=float((eval_adapter(new,**inputs,noise=noise)-eval_adapter(loaded,**inputs,noise=noise)).abs().max());assert error<=2e-6
        result[label]=dict(executed=True,cases=cases,roundtrip_error=error)
        del old,new,loaded
    write(a.run/'public_contract_errors.json',result)
    # Full architecture, real data, dropout enabled, exact same schedule on GPU.
    cfg=TrainConfig(max_steps=8,checkpoint_interval=4,eval_interval=4,diagnostic_interval=4,fixed_checkpoints=(4,8))
    m=prepare(a.run/'resume_smoke',cfg);records,eps,_=schedule(pool,cfg,32)
    results={}
    for arm in ('C0','C1','C2'):
        data=RealData(m,cfg,arm,a.device)
        full,mf,of=train(a.run/'resume_smoke'/('full_'+arm),cfg,arm,records,eps,data,m,a.device)
        stop,ms,os=train(a.run/'resume_smoke'/('split_'+arm),cfg,arm,records,eps,data,m,a.device,stop_after=4)
        del ms,os
        resumed,mr,orr=train(a.run/'resume_smoke'/('split_'+arm),cfg,arm,records,eps,data,m,a.device,resume=stop['checkpoints'][-1]['path'])
        error=max(float((v-mr.state_dict()[k]).abs().max()) for k,v in mf.state_dict().items())
        assert state_hash(mf)==state_hash(mr) and error==0
        results[arm]=dict(parameter_max_error=error,stopped_completed=stop['completed'],requested=cfg.max_steps,
                          actual=resumed['actual_optimizer_steps'],full=full,resumed=resumed)
        del data,mf,of,mr,orr;torch.cuda.empty_cache()
    write(a.run/'resume_gpu_results.json',results)


if __name__=='__main__':main()
