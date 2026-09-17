"""Three phases: source-only prediction, sealed selection, independent evaluation."""
from pathlib import Path
import json, time
import numpy as np
import torch
from reaction_reward.common import ROOT, OUT as OLD, read, write, sha
from reaction_reward.data import Data
from reaction_reward.quality_v2.model import Judge
from .selectors import select
BASE = ROOT/'runs/reaction_reward/quality_v2'
OUT = ROOT/'runs/reaction_reward/vector_selection_v1'
ARMS = ['A-repaired-RMGen', 'B-quality-heads']
GENS = ['P2', 'N0', 'N1']

def predict(windows):
    data=Data(); torch.set_num_threads(3)
    for arm in ARMS:
        ck=read(BASE/'training'/arm/'SELECTION.json'); assert ck['step']==500
        model=Judge().cuda().eval(); model.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['model']); model.requires_grad_(False)
        for wi,w in enumerate(windows):
            for gen in GENS:
                path=OUT/'predictions'/arm/(w['id']+'_'+gen+'.json')
                if path.exists(): continue
                # Source-only loader deliberately does not call Data.target or Data.batch.
                x,t=data.source(w['source'],w['start'],w['n'])
                y=np.load(OLD/'bank'/w['id']/(gen+'.npy'))
                yn=data.normalize(y,'facial-attributes/listener')
                assert y.shape == (16,w['n'],25)
                values=[]
                with torch.inference_mode():
                    for j in range(0,16,4):
                        tx=torch.tensor(np.repeat(x[None],4,0),device='cuda',dtype=torch.float32)
                        ty=torch.tensor(yn[j:j+4],device='cuda',dtype=torch.float32)
                        tt=torch.tensor(np.repeat(t[None],4,0),device='cuda',dtype=torch.float32)
                        mask=torch.ones(tt.shape,device='cuda',dtype=torch.bool)
                        values.extend(model(tx,ty,tt,tt,mask,mask).cpu().tolist())
                write(path,dict(values=values,candidate_sha256=sha(OLD/'bank'/w['id']/(gen+'.npy')),checkpoint_sha256=sha(ck['path']) if wi==0 and gen=='P2' else None))
            write(OUT/'status.json',dict(phase='frozen inference',arm=arm,windows=wi+1,total=len(windows),time=time.time()))
            print(arm,wi+1,len(windows),flush=True)
        del model; torch.cuda.empty_cache()

def selection(windows,sc):
    rows=[]
    for w in windows:
        for gen in GENS:
            v=[read(OUT/'predictions'/a/(w['id']+'_'+gen+'.json'))['values'] for a in ARMS]
            y=np.load(OLD/'bank'/w['id']/(gen+'.npy'))
            rows.append(dict(window=w['id'],group=w['group'],generator=gen,n=w['n'],selection=select(y,*v,sc['mean'],sc['std'])))
    write(OUT/'SEALED_SELECTIONS.json',rows)
    write(OUT/'SELECTION_SEAL.json',dict(sha256=sha(OUT/'SEALED_SELECTIONS.json'),time=time.time(),target_scores_read=False))

def metrics(y,c,d,ids):
    flat=y[ids].reshape(10,-1).astype(np.float64)
    # Identical ordered-pair / 90 definition to previous reports.
    smse=float(2 * 10 / 9 * np.var(flat,axis=0).mean())
    return dict(sum_best_CCC=float(c[ids].sum()),sum_best_native_DTW=float(d[ids].sum()),S_MSE=smse)

def evaluate(windows,sc):
    sealed=read(OUT/'SEALED_SELECTIONS.json'); assert sha(OUT/'SEALED_SELECTIONS.json')==read(OUT/'SELECTION_SEAL.json')['sha256']
    details=[]; lookup={}
    for row in sealed:
        wid,gen=row['window'],row['generator']; y=np.load(OLD/'bank'/wid/(gen+'.npy'))
        meta=read(OLD/'bank'/wid/(gen+'_scores.json'));c=np.array(meta['C']).max(1);d=np.array(meta['D']).min(1);logd=np.log1p(d/np.sqrt(row['n']))
        p2=read(OLD/'bank'/wid/'P2_scores.json'); py=np.load(OLD/'bank'/wid/'P2.npy');pc=np.array(p2['C']).max(1);pd=np.array(p2['D']).min(1)
        fixed=metrics(py,pc,pd,list(range(10)));first=metrics(y,c,d,list(range(10)))
        for method,s in row['selection'].items():
            val=metrics(y,c,d,s['selected']);r=dict(window=wid,group=row['group'],generator=gen,method=method,metrics=val,
                delta_first10={k:val[k]-first[k] for k in val},delta_fixed_P2_first10={k:val[k]-fixed[k] for k in val},fallback=s['fallback'],fallback_slots=s.get('fallback_slots',0))
            r['selected_fail_P2_quality']=int(((c[s['selected']]<pc[:10].mean())|(d[s['selected']]>pd[:10].mean())).sum())
            if method.startswith('vector-'):
                threshold=np.array(sc['mean']) if method=='vector-train' else np.array([c[:10].mean(),logd[:10].mean()])
                true_gate=(c>=threshold[0])&(logd<=threshold[1]);pg=np.array(s['predicted_gate']);accepted=s['gate_accepted']
                r.update(gate_pass=int(pg.sum()),false_pass=int((pg&~true_gate).sum()),accepted_count=len(accepted),accepted_false=int((~true_gate[accepted]).sum()))
            details.append(r)
        lookup[wid,gen]=True
    write(OUT/'PER_WINDOW.json',details)
    def aggregate(rs):
        result=dict(windows=len(rs),date_groups=len({r['group'] for r in rs}),fallback_rate=float(np.mean([r['fallback'] for r in rs])),fallback_slot_rate=float(np.mean([r['fallback_slots']/10 for r in rs])),selected_fail_P2_quality_rate=sum(r['selected_fail_P2_quality'] for r in rs)/(10*len(rs)))
        for key in ['metrics','delta_first10','delta_fixed_P2_first10']:
            result[key]={k:float(np.mean([r[key][k] for r in rs])) for k in rs[0][key]}
        for key in ['delta_first10','delta_fixed_P2_first10']:
            result[key]['joint_quality_nonworse_rate']=float(np.mean([r[key]['sum_best_CCC']>=-1e-9 and r[key]['sum_best_native_DTW']<=1e-9 for r in rs]))
        if 'gate_pass' in rs[0]:
            for num,den,name in [('false_pass','gate_pass','false_pass_rate'),('accepted_false','accepted_count','accepted_false_pass_rate')]:
                n=sum(r[num] for r in rs); d=sum(r[den] for r in rs);result[name]=dict(numerator=n,denominator=d,rate=n/d if d else None)
        return result
    summary={gen:{method:aggregate([r for r in details if r['generator']==gen and r['method']==method]) for method in sealed[0]['selection']} for gen in GENS}
    groups={g:{gen:{method:aggregate([r for r in details if r['group']==g and r['generator']==gen and r['method']==method]) for method in sealed[0]['selection']} for gen in GENS} for g in sorted({r['group'] for r in details})}
    write(OUT/'SUMMARY.json',summary);write(OUT/'PER_DATE.json',groups)
    # Generated preference readout comparison uses the same held-fixed checkpoint predictions.
    records=[r for r in read(BASE/'RECORDS.json') if r['split']=='RM_audit' and r['family']=='generated' and r['state']=='PREFERRED' and r['weight']>0]
    pair=[]
    for r in records:
        vals=[]
        for side in ['A','B']:
            z=[]
            for arm in ARMS:
                gen=Path(r[side]['generated']).stem
                z.append(np.array(read(OUT/'predictions'/arm/(r['window']+'_'+gen+'.json'))['values'])[r[side]['candidate']])
            vals.append([z[0][0]+z[0][1],z[1][0]+z[1][1],z[1][2]-z[1][3]])
        pair.append(dict(window=r['window'],group=r['group'],unseen=r['unseen'],difference=(np.array(vals[0])-vals[1]).tolist()))
    write(OUT/'READOUT_PAIRS.json',pair)
    comparison={}
    for name,subset in [('all',pair),('unseen_N1',[r for r in pair if r['unseen']])]:
        z=np.array([r['difference'] for r in subset]);comparison[name]=dict(n=len(subset),readouts={key:dict(accuracy=float((z[:,i]>0).mean()),raw_NLL=float(np.logaddexp(0,-z[:,i]).mean())) for i,key in enumerate(['A-context-temporal','B-context-temporal','B-quality-CD'])})
    write(OUT/'READOUT_COMPARISON.json',comparison)
    lines=['# Frozen RM vector selection','', 'Reused RM_audit development diagnosis; 96 windows / 3 recording-date groups. Native-window best-of16 selection, not full-recording FRC/FRD or native K10 inference. No training or generator changes.','', '|Generator|Selection|CCC sum ↑|Native DTW sum ↓|S-MSE ↑|Fallback|','|---|---|---:|---:|---:|---:|']
    for gen,methods in summary.items():
        for method,r in methods.items():
            m=r['metrics'];lines.append(f"|{gen}|{method}|{m['sum_best_CCC']:.6f}|{m['sum_best_native_DTW']:.6f}|{m['S_MSE']:.6f}|{r['fallback_rate']:.1%}|")
    lines += ['', 'Readout comparisons: `READOUT_COMPARISON.json`. Quality deltas versus same-pool first10 and fixed P2 first10, false-pass numerator/denominator, fallback slots: `SUMMARY.json`; group results: `PER_DATE.json`.', '', 'Predicted gates separately compare CCC and transformed distance: fixed RM_fit means, or same-pool predicted first10 means. Native-channel squared-distance greedy selection; quality-ranked fill when fewer than ten pass. A fallback is NOT quality-approved. False pass means predicted dual pass but failure of at least one corresponding independently computed true-quality threshold; it is not human preference error.', '', 'Same-pool gates do not restore P2 quality. Fixed P2 first10 comparisons are explicitly separate. There is no audit-based threshold fitting, no calibrated probability claim, and no independent confirmation from this reused split.', '', 'Inference reads source feature arrays/source PTS and candidate arrays only, plus frozen TRAIN normalization. Candidate time uses the source native grid, not listener PTS. Historical candidate pools and windows were previously constructed using targets; this is a fixed-pool diagnostic, not a proof of deployment sampling.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'completion.json',dict(complete=True,training=False,generator_updated=False,RL_started=False,time=time.time()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists(): raise RuntimeError('completed output exists; do not overwrite')
    windows=[{k:w[k] for k in ['id','source','group','split','start','n']} for w in read(OLD/'WINDOWS.json') if w['split']=='RM_audit']
    sc=read(BASE/'QUALITY_SCALE.json'); assert sc['fit_split']=='RM_fit'
    protocol=dict(checkpoints={a:dict(path=read(BASE/'training'/a/'SELECTION.json')['path'],sha256=sha(read(BASE/'training'/a/'SELECTION.json')['path'])) for a in ARMS},quality_scale=sc,quality_scale_sha256=sha(BASE/'QUALITY_SCALE.json'),scope='reused development RM_audit',K=10,pool=16,gate_policies=['RM_fit mean CCC and mean log1p(D/sqrt(n))','same-pool predicted first10 mean CCC and logD'],fallback='fill by frozen B zC-zD ranking; report separately',selection='highest zC-zD eligible seed; greedy mean native squared distance; ties lowest index',time_axis='source native PTS for source and generated candidate',forbidden=['target-informed selection','training','generator updates','Q0/Q1 resume','automatic push'])
    if not (OUT/'PROTOCOL.json').exists():write(OUT/'PROTOCOL.json',protocol)
    else:assert read(OUT/'PROTOCOL.json')==protocol
    predict(windows);selection(windows,sc);evaluate(windows,sc)
if __name__=='__main__':main()
