"""Weak-answer retention and chunk-aggregated full-recording consistency."""
import numpy as np
import torch
from .common import *
from .data import Data
from .model import Judge
from .train import ARMS

def main():
 torch.set_num_threads(2);data=Data();windows=[w for w in read(OUT/'WINDOWS.json') if w['split']=='RM_audit'];evidence=read(OUT/'REAL_EVIDENCE.json');results={}
 for arm in ARMS:
  sel=read(OUT/'training'/arm/'SELECTION.json');m=Judge().cuda();m.load_state_dict(torch.load(sel['selected_checkpoint'],map_location='cpu',weights_only=False)['model']);m.eval();cal=read(OUT/'training'/arm/f"cal_{sel['selected_step']:06d}.json");neg=[-r['difference'] for r in cal if r['family']=='base_context' and r['weight']>0];threshold=float(np.quantile(neg,.95)) if neg else None;answers=[];consistency=[]
  def value(source,start,n,target):
   row=dict(source=source,start=start,n=n,A=target)
   with torch.no_grad():return m(*data.batch([row],'A'))[0].cpu().numpy()
  for w in windows:
   ref=value(w['source'],w['start'],w['n'],w['targets'][0]);deltas=[]
   for target in w['targets'][1:]:deltas.append(float(value(w['source'],w['start'],w['n'],target)[0]-ref[0]))
   answers.append(dict(window=w['id'],group=w['group'],deltas=deltas,accepted=sum(x>=threshold for x in deltas) if threshold is not None else None,total=len(deltas),reference='same-source observed paired score; alternatives weak global only'))
   r=data.rows[w['source']];N=min(len(r['pts']['speaker']),len(r['pts']['listener']));chunks=[];weights=[]
   for s in range(0,N,750):
    n=min(750,N-s)
    if n<2:continue
    chunks.append(value(w['source'],s,n,dict(id=w['source'],start=s)).tolist());weights.append(n)
   full=np.average(chunks,axis=0,weights=weights);consistency.append(dict(window=w['id'],selected_window_score=ref.tolist(),chunk_aggregated_full_recording_score=full.tolist(),chunks=len(chunks),frames=N))
   print('DIAGNOSTIC',arm,w['id'],flush=True)
  a=np.array([x['selected_window_score'] for x in consistency]);b=np.array([x['chunk_aggregated_full_recording_score'] for x in consistency]);result=dict(acceptance_threshold=threshold,calibration='RM_cal 95th percentile mismatched-minus-paired context scores; engineering weak-support retention, not calibrated correctness',weak_answers=answers,recording_consistency=consistency,score_correlations=[float(np.corrcoef(a[:,i],b[:,i])[0,1]) if a[:,i].std()>0 and b[:,i].std()>0 else None for i in range(2)],full_score_definition='length-weighted native <=750-frame windows, not a single full-sequence attention forward or official FRC')
  write(OUT/'audit'/arm/'retention_and_recording.json',result);results[arm]=dict(weak_total=sum(x['total'] for x in answers),weak_accepted=sum(x['accepted'] or 0 for x in answers))
 write(OUT/'MULTIANSWER_RETENTION.json',results)
if __name__=='__main__':main()
