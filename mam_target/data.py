import hashlib,random
import torch
import torch.nn.functional as F
from hirp.data_phase25 import RealData

class TaskData(RealData):
    def task_batch(self,record):
        batch,refs=self.batch(record);targets=[];lens=[];ids=[]
        for i,clip in enumerate(batch['clip_id']):
            paired=clip.replace('speaker/','listener/',1)
            pool=[r for r in self.pool.references[record['session_id']] if r!=paired]
            if not pool:pool=[paired]
            seed=int.from_bytes(hashlib.sha256(f"123|{record['step']}|{i}|{clip}".encode()).digest()[:8],'big')
            rng=random.Random(seed);chosen=rng.sample(pool,3) if len(pool)>=3 else rng.choices(pool,k=3)
            start=int(batch['crop_start'][i]);total=int(batch['source_total_length'][i]);n=int(batch['source_lengths'][i])
            ts=[batch['paired_target'][i]];ls=[int(batch['pair_lengths'][i])]
            for ref in chosen:
                raw=self.pool.dataset._load(self.pool.reference_paths[ref])
                # Explicit TRAIN weak-label alignment: full trajectory linear time warp,
                # then SAME source crop. Not official Processor nor causal frame pairing.
                aligned=F.interpolate(raw.T[None],size=total,mode='linear',align_corners=True)[0].T
                crop=raw.new_zeros(batch['paired_target'].shape[1],25);crop[:n]=aligned[start:start+n]
                ts.append(crop.to(self.device));ls.append(n)
            targets.append(torch.stack(ts));lens.append(ls);ids.append([paired]+chosen)
        return batch,refs,torch.stack(targets),torch.tensor(lens,device=self.device),ids
