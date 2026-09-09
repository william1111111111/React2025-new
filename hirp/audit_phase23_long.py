"""No-training integration tests at long input shapes on real trained HiRP."""
import argparse,json
from pathlib import Path
import torch
from .phase22 import load_checkpoint,eval_adapter
from .phase23_long import export_full
from .train_phase22 import write
from .train_phase21 import configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/phase23/tradeoff_v1'));p.add_argument('--device',default='cuda:7');a=p.parse_args();configure()
    reused=json.loads((a.root/'reused_checkpoints.json').read_text());model,_=load_checkpoint(reused[0]['checkpoint']['path'],a.device)
    results=[];g=torch.Generator().manual_seed(23001)
    for t in (128,256,750):
        b=2;inputs=dict(speaker_audio=torch.randn(b,t,768,generator=g).to(a.device),speaker_emotion=torch.randn(b,t,25,generator=g).to(a.device),speaker_3dmm=torch.randn(b,t,58,generator=g).to(a.device),lengths=torch.tensor([t,37],device=a.device))
        noise=torch.randn(b,10,32,generator=g).to(a.device)
        with torch.no_grad():x=model(**inputs,sample_count=10,noise=noise)
        sample=model.sample(**inputs,sample_count=10,noise=noise);adapter=eval_adapter(model,**inputs,noise=noise)
        chunk=torch.cat([eval_adapter(model,**inputs,noise=noise[:,i:i+3]) for i in range(0,10,3)],1)
        loaded,_=load_checkpoint(reused[0]['checkpoint']['path'],a.device);rt=eval_adapter(loaded,**inputs,noise=noise)
        valid=torch.arange(t,device=a.device)[None]<inputs['lengths'][:,None];v=x[valid[:,None].expand(b,10,t)]
        assert x.shape==(b,10,t,25) and x[~valid[:,None].expand(b,10,t)].count_nonzero()==0
        assert (v[:,:15]>=0).all() and (v[:,:15]<=1).all() and (v[:,15:17].abs()<=1).all()
        torch.testing.assert_close(v[:,17:].sum(-1),torch.ones(len(v),device=a.device))
        errors={name:float((y-x).abs().max()) for name,y in [('sample',sample),('adapter',adapter),('loaded',rt),('candidate_chunks',chunk)]}
        assert all(e<3e-6 for e in errors.values()),errors
        results.append(dict(T=t,lengths=[t,37],shape=list(x.shape),max_errors=errors));del loaded
    n=1001;streams=dict(speaker_audio=torch.randn(n,768,generator=g).to(a.device),speaker_emotion=torch.randn(n,25,generator=g).to(a.device),speaker_3dmm=torch.randn(n,58,generator=g).to(a.device));noise=torch.randn(10,32,generator=g).to(a.device)
    a1=export_full(model,**streams,source_length=n,noise=noise,candidate_chunk=10)
    a2=export_full(model,**streams,source_length=n,noise=noise,candidate_chunk=2)
    error=float((a1-a2).abs().max());assert error<3e-6
    with torch.no_grad():direct=eval_adapter(model,**{k:v[None] for k,v in streams.items()},lengths=torch.tensor([n],device=a.device),noise=noise[None])[0].cpu()
    write(a.root/'long_interface_audit.json',dict(shape_tests=results,reassembly_shape=list(a1.shape),candidate_chunk_max_error=error,
        time_chunk_vs_full_context_max_difference=float((a1-direct).abs().max()),time_chunk_equivalence_claimed=False,
        global_latent='same [10,32] noise used for every time chunk',prediction_quality_claimed=False))


if __name__=='__main__':main()
