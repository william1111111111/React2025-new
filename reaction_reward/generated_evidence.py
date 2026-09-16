import random,collections
import numpy as np
from .common import *
def main():
 rng=random.Random(123);result=[]
 for w in read(OUT/'WINDOWS.json'):
  names=['P2','N0']+(['N1'] if w['split']=='RM_audit' else []);scores={k:read(OUT/'bank'/w['id']/(k+'_scores.json')) for k in names};dref=float(np.array(scores['P2']['D']).min(1).mean());same=[];cross=[];unknown=[]
  for ia,ma in enumerate(names):
   for mb in names[ia:]:
    for i in range(16):
     for j in range(i+1 if ma==mb else 0,16):
      ca,cb=max(scores[ma]['C'][i]),max(scores[mb]['C'][j]);da,db=min(scores[ma]['D'][i]),min(scores[mb]['D'][j]);win=ca>=cb+.01 and da<=db-.05*max(dref,1e-8);lose=cb>=ca+.01 and db<=da-.05*max(dref,1e-8)
      a=dict(id=w['source'],start=w['start'],generated=str(OUT/'bank'/w['id']/(ma+'.npy')),candidate=i);b=dict(id=w['source'],start=w['start'],generated=str(OUT/'bank'/w['id']/(mb+'.npy')),candidate=j)
      if lose:a,b=b,a
      r=dict(family='generated',source=w['source'],start=w['start'],n=w['n'],A=a,B=b,split=w['split'],group=w['group'],weight=1. if win or lose else 0.,state='PREFERRED' if win or lose else 'UNKNOWN',window=w['id'],generators=[ma,mb],unseen='N1' in [ma,mb],D_ref=dref)
      (same if ma==mb else cross).append(r) if win or lose else unknown.append(r)
  rng.shuffle(same);rng.shuffle(cross);rng.shuffle(unknown)
  # At least half within-source-generator comparisons, max8 labels/window.
  selected=same[:4];selected+=cross[:min(4,len(selected))];selected+=same[4:4+max(0,8-len(selected))]
  result.extend(selected[:8]+unknown[:8])
 write(OUT/'GENERATED_EVIDENCE.json',result);print('GENERATED PREFERENCES',dict(collections.Counter(x['split']+'_'+x['state'] for x in result)),flush=True)
if __name__=='__main__':main()
