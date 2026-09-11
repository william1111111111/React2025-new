"""Single-seed fixed-bank ES curves; T750 and historical T128 stay separate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .phase22 import load_checkpoint
from .evaluate_phase22 import evaluate_model,extra_shuffles
from .group_features import DescriptorScaler,reaction_descriptor
from .session_data import SessionPopulation
from .data_phase25 import RawCachedDataset,reference_crop
from .phase24 import verify_files,normalization,dependencies
from .phase23_cache import cached,save_cached
from .phase15_audit import sha256_file,canonical_hash
from .train_phase22 import write
from .train_phase21 import configure

ROOT=Path('runs/phase25/timescale_v1');OUT=ROOT/'final_seed123_v1'
PLAN=Path('runs/phase22/replicated_v1/evaluation_plan/manifest.json')

class SourceSerialSampler:
    """Engineering source batch1, K unchanged and every sample identity retained."""
    def __init__(self,model):self.model=model
    def sample(self, speaker_audio,speaker_emotion,speaker_3dmm,lengths,sample_count,noise):
        return torch.cat([self.model.sample(speaker_audio[i:i+1],speaker_emotion[i:i+1],speaker_3dmm[i:i+1],lengths[i:i+1],sample_count=sample_count,noise=noise[i:i+1]) for i in range(len(lengths))])

def population(T,plan):
    pool=SessionPopulation('data','val',T)
    if T==750:pool.dataset=RawCachedDataset('data','val',750,crop_mode='random',seed=25001)
    items=[]
    for occurrence,i in enumerate(plan['indices']):
        pool.dataset.set_epoch(occurrence);items.append(pool.dataset[i])
    batch=default_collate(items);assert batch['clip_id']==plan['clip_ids']
    sp=ROOT/'descriptor_stats_T750.json' if T==750 else Path('runs/phase2/descriptor_stats.json')
    scaler=DescriptorScaler.load(sp);refs={};rc=[]
    for session in sorted(set(batch['session_id'])):
        values=[];valids=[]
        for ref in pool.references[session]:
            if T==750:y,n,start=reference_crop(pool,ref,25001,0)
            else:
                item=pool.load_reference(ref);y,n,start=item['reaction'],item['length'],item['crop_start']
            f,v=scaler(*reaction_descriptor(y,n));values.append(f);valids.append(v)
            rc.append(dict(reference_id=ref,path=str(pool.reference_paths[ref]),sha256=sha256_file(pool.reference_paths[ref]),start=start,length=int(n)))
        refs[session]=(torch.stack(values),torch.stack(valids))
    crops=[dict(clip_id=batch['clip_id'][i],start=int(batch['crop_start'][i]),source_length=int(batch['source_lengths'][i]),pair_length=int(batch['pair_lengths'][i])) for i in range(80)]
    return batch,refs,scaler,sp,dict(T=T,source_crops=crops,reference_crops=rc,scaler_path=str(sp),scaler_sha256=sha256_file(sp),crop_rule='uniform source-only and independent reference seed25001' if T==750 else 'historical center crop')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args();configure()
    protocol=json.loads((OUT/'es_protocol.json').read_text());plan=json.loads(PLAN.read_text());normalization(plan);verify_files(plan['data_files']+plan['normalization']+plan['banks'])
    snapshots=dependencies();out=OUT/'distribution';out.mkdir(exist_ok=True)
    for T in (750,128):
        batch,refs,scaler,sp,pop=population(T,plan)
        if T==750:
            train=json.loads((ROOT/'seed_123/manifest.json').read_text());assert sha256_file(sp)==train['scales']['descriptor_scaler_sha256']
        else:assert sha256_file(sp)==plan['scaler_sha256']
        popfile=out/f'population_T{T}.json'
        if popfile.exists():assert json.loads(popfile.read_text())==pop
        else:write(popfile,pop)
        for arm in ('C0','C1','C2'):
            for step in (128,500,1000,2000,4000,6000):
                cp=sorted((ROOT/'seed_123'/arm).glob(f'attempt_*/checkpoints/step_{step:06d}.pt'))[0]
                model,meta=load_checkpoint(cp,a.device);assert meta['arm']==arm and meta['global_step']==step and meta['train_config']['init_seed']==123
                for bi in ([0,1] if step in (2000,6000) else [0]):
                    bank_info=plan['banks'][bi];bank=torch.from_numpy(np.load(bank_info['path']))[:,:32]
                    identity=dict(version='phase25-single-seed-es-v1',T=T,K=32,bank=bank_info,checkpoint_sha256=sha256_file(cp),arm=arm,step=step,seed=123,population_sha256=sha256_file(popfile),plan_sha256=sha256_file(PLAN),protocol_sha256=sha256_file(OUT/'es_protocol.json'),dependencies=snapshots,data_fingerprints=plan['data_files'],normalization=plan['normalization'],model_scales=meta['scales'],source_batch=1)
                    path=out/f'{arm}_step{step}_T{T}_bank{bi}_K32.json'
                    if cached(path,identity) is not None:continue
                    values=evaluate_model(SourceSerialSampler(model),batch,refs,scaler,scaler.to(a.device),bank,torch.tensor(plan['permutations'][0]['donors']),a.device,32)
                    predictions=values.pop('all_predictions')
                    if step in (2000,6000):values['extra_shuffles']=extra_shuffles(predictions,batch,plan['permutations'])
                    if T==750 and step==6000 and bi==0:
                        path_np=out/f'{arm}_step6000_T750_bank0_predictions.npy'
                        with path_np.open('xb') as f:np.save(f,predictions.numpy())
                        values['predictions']=dict(path=str(path_np),sha256=sha256_file(path_np))
                    values.update(arm=arm,step=step,T=T,bank=bi,K=32,seed=123)
                    save_cached(path,identity,values);print(path.name,values['aggregate']['conditional']['loss'],values['aggregate']['marginal']['loss'],flush=True)
                del model;torch.cuda.empty_cache()
    write(out/'completed.json',dict(completed=True,cases=48,T=[750,128],K=32,seed=123,new_noise_banks=0))

if __name__=='__main__':main()
