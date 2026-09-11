import torch,numpy as np
from mam_target.model import load_checkpoint
from mam_target.losses import pair_costs
from mam_staged.losses import quality_proxies,dispersions,losses
from mam_staged.params import partition
from mam_staged.numerics import consistent_attention
from mam_staged.common import ROOT,write,resources
from hirp.paired_data import paired_model_inputs
from hirp.train_phase21 import configure

def positive_median(values,floor):
    x=np.asarray(values);x=x[np.isfinite(x)&(x>0)];return max(floor,float(np.median(x))) if len(x) else floor

def main():
    configure();torch.use_deterministic_algorithms(True);m,records,noises,data=resources('cuda:0');parent,_=load_checkpoint(m['parent_checkpoint'],'cuda:0');consistent_attention(parent);parent.requires_grad_(False);parent.eval();cache=[];cs=[];ds=[];refs=[[] for _ in range(6)];skipped=0;nearzero=0
    for rec,noise in zip(records[:16],noises[:16]):
        b,_,y,n,ids=data.task_batch(rec);noise=noise.to('cuda:0')
        with torch.no_grad():
            p=parent(**paired_model_inputs(b),sample_count=4,noise=noise);c,d=pair_costs(p,y,b['source_lengths'],n,m['weights']['grid']);qc,qd=quality_proxies(c,d);cs+=qc.tolist();ds+=qd.tolist();_,_,dr,valid=dispersions(p,p,y,b['source_lengths'],n,ids)
        for j in range(6):refs[j]+=dr[:,j][valid[:,j]].tolist()
        skipped+=int((~valid).sum());nearzero+=int(((dr<=1e-12)&valid).sum())
        cache.append((rec,noise.cpu(),p.cpu(),c.cpu(),d.cpu()));print('SCALES',rec['step'],flush=True)
    cal=dict(s_C=positive_median(cs,1e-4),s_D=positive_median(ds,1e-4),epsilon=[max(1e-6,.01*positive_median(v,0)) for v in refs],skipped_components=skipped,nearzero_reference_components=nearzero,calibration_batches=16,reference_positive_counts=[sum(vv>0 for vv in v) for v in refs],lambda_max=0.)
    student,_=load_checkpoint(m['parent_checkpoint'],'cuda:0');consistent_attention(student);student.eval();phi=[p for _,p in partition(student)['stochastic']];ratios=[];rows=[]
    for rec,noise,p,c0,d0 in cache:
        b,_,y,n,ids=data.task_batch(rec);q=student(**paired_model_inputs(b),sample_count=4,noise=noise.to('cuda:0'))
        v,_=losses(q,p.to('cuda:0'),b,y,n,ids,m['weights'],cal,500,'S1_joint',(c0.to('cuda:0'),d0.to('cuda:0')),alpha_override=.25)
        norms=[]
        for term in ('quality','disp'):
            grad=torch.autograd.grad(v['r2'] if term=='quality' else v[term],phi,retain_graph=True,allow_unused=True);norms.append(float(sum(g.square().sum() for g in grad if g is not None).sqrt()))
        if norms[1]>1e-12:ratios.append(.2*norms[0]/norms[1])
        rows.append(dict(global_step=rec['step'],grad_quality_phi=norms[0],grad_disp_phi=norms[1],guard_C=float(v['guard_C']),guard_D=float(v['guard_D']),disp=float(v['disp'])))
        error=float((q.detach()-p.to('cuda:0')).abs().max());rows[-1]['parent_student_max_error']=error
        assert error<2e-6 and max(float(v['guard_C']),float(v['guard_D']))<1e-5, rows[-1]
        print('GRAD',rec['step'],norms,flush=True)
    if not ratios:raise RuntimeError('all dispersion gradients near zero; inspect sensitivity, do not launch')
    cal.update(lambda_max=min(10.,float(np.median(ratios))),ignored_gradient_batches=16-len(ratios),gradient_rows=rows,phi_names=[n for n,_ in partition(student)['stochastic']],rule='median .2*quality_grad/disp_grad on TRAIN16 at alpha .25; cap10; no DEV',epsilon_labels=m['epsilon_order'],parent_coincidence_policy='At the common parent L_guard and its ReLU(0) gradient are analytically zero; use original R2 gradient for calibration, record independently evaluated FP32 discrepancy; tolerance 2e-6 output/1e-5 guard')
    write(ROOT/'calibration.json',cal);write(ROOT/'parameter_groups.json',{k:[dict(name=n,numel=p.numel()) for n,p in v] for k,v in partition(student).items()});print(cal['lambda_max'],flush=True)
if __name__=='__main__':main()
