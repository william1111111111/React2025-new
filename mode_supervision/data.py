"""Whole recordings for plans, generation-aligned blocks for decoder losses."""
import hashlib,json,random
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F
from hirp.session_data import SessionPopulation
from .prepare import OUT
from .plans import recording

class ModeData:
    def __init__(self, base, device='cuda:0'):
        self.base,self.device=base,device
        self.rows=json.loads((OUT/'cohort.json').read_text())
        self.pool=SessionPopulation('data','train',750)
        self.ds=self.pool.dataset

    @lru_cache(maxsize=32)
    def source(self,index):
        r=self.rows[index];rel=Path(r['clip_id']+'.npy');root=Path('data/train')
        values=[self.ds._load(root/f/rel) for f in ('audio-features','facial-attributes','coefficients')]
        values[2]=values[2].reshape(-1,58);values[2]=(values[2]-self.ds.mean)/self.ds.std
        return [v[:r['source_length']] for v in values]

    @lru_cache(maxsize=256)
    def target(self,ref):return self.ds._load(self.pool.reference_paths[ref])

    def refs(self,rec,i):
        clip=self.rows[rec['source_indices'][i]]['clip_id'];paired=clip.replace('speaker/','listener/',1)
        pool=[r for r in self.pool.references[rec['session_id']] if r!=paired] or [paired]
        seed=int.from_bytes(hashlib.sha256(f"123|{rec['step']}|{i}|{clip}".encode()).digest()[:8],'big')
        rng=random.Random(seed)
        return [paired]+(rng.sample(pool,3) if len(pool)>=3 else rng.choices(pool,k=3))

    def aligned(self,index,ref,paired=False):
        raw=self.target(ref);n=self.rows[index]['source_length']
        return raw[:n] if paired else F.interpolate(raw.T[None],size=n,mode='linear',align_corners=True)[0].T

    def plan(self,index,y):
        pts=self.rows[index]['frame_pts'][:len(y)]
        with torch.no_grad():a,m=recording(self.base.transform(y.to(self.device)),pts)
        blocks=(self.rows[index]['source_length']+749)//750
        return F.pad(a,(0,0,0,blocks-len(a))),F.pad(m,(0,0,0,blocks-len(m)))

    @torch.no_grad()
    def context(self,index):
        path=OUT/'source_context'/f'{index:04d}.pt'
        if path.exists():return torch.load(path,map_location=self.device,weights_only=True)
        r=self.rows[index];values=self.source(index);hs=[];positions=[]
        for start in range(0,r['source_length'],750):
            n=min(750,r['source_length']-start)
            xs=[F.pad(v[start:start+n],(0,0,0,750-n))[None].to(self.device) for v in values]
            h,m=self.base.condition(*xs,torch.tensor([n],device=self.device))
            hs.append((h*m[...,None]).sum(1)[0]/n)
            p=r['frame_pts'];positions.append([p[start]/30,(p[start+n-1]-p[start])/30])
        result=dict(context=torch.stack(hs).cpu(),positions=torch.tensor(positions),mask=torch.stack([torch.arange(4)[:,None].expand(4,24)<min(4,r['source_length']-s) for s in range(0,r['source_length'],750)]).flatten(1))
        path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(path)
        return {k:v.to(self.device) for k,v in result.items()}

    def batch(self,rec,with_context=True):
        items=[];plans=[];masks=[];contexts=[];positions=[];prior_masks=[];all_ids=[];pts=[]
        maxblocks=max((self.rows[i]['source_length']+749)//750 for i in rec['source_indices'])
        for i,index in enumerate(rec['source_indices']):
            row=self.rows[index];start=rec['block_indices'][i]*750;n=min(750,row['source_length']-start)
            xs=self.source(index);ids=self.refs(rec,i);ys=[self.aligned(index,ref,j==0) for j,ref in enumerate(ids)]
            lens=[max(0,min(n,len(y)-start)) for y in ys]
            if lens[0]<1:raise ValueError('source block without paired overlap; explicit schedule repair required')
            crops=[F.pad(y[start:start+nn],(0,0,0,750-nn)) for y,nn in zip(ys,lens)]
            item=dict(speaker_audio=F.pad(xs[0][start:start+n],(0,0,0,750-n)),speaker_emotion=F.pad(xs[1][start:start+n],(0,0,0,750-n)),speaker_3dmm=F.pad(xs[2][start:start+n],(0,0,0,750-n)),source_lengths=torch.tensor(n),paired_target=crops[0],pair_lengths=torch.tensor(lens[0]),crop_start=torch.tensor(start),_targets=torch.stack(crops),_target_lengths=torch.tensor(lens))
            slot=rec['target_slots'][i];a,m=self.plan(index,ys[slot]);plans.append(F.pad(a,(0,0,0,maxblocks-len(a))));masks.append(F.pad(m,(0,0,0,maxblocks-len(m))))
            if with_context:
                c=self.context(index);contexts.append(F.pad(c['context'],(0,0,0,maxblocks-len(c['context']))));positions.append(F.pad(c['positions'],(0,0,0,maxblocks-len(c['positions']))));prior_masks.append(F.pad(c['mask'],(0,0,0,maxblocks-len(c['mask']))))
            items.append(item);all_ids.append(ids);pts.append(row['frame_pts'][start:start+n])
        batch={k:torch.stack([r[k] for r in items]).to(self.device) for k in items[0]}
        index=torch.arange(len(items),device=self.device);slots=torch.tensor(rec['target_slots'],device=self.device)
        return dict(batch=batch,y=batch['_targets'][index,slots],lengths=batch['_target_lengths'][index,slots],plan=torch.stack(plans),mask=torch.stack(masks),context=torch.stack(contexts) if with_context else None,positions=torch.stack(positions) if with_context else None,prior_mask=torch.stack(prior_masks) if with_context else None,ids=all_ids,pts=pts,block_index=torch.tensor(rec['block_indices'],device=self.device))
