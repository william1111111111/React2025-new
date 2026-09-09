"""Split-local unique populations and source-uniform session occurrence schedules."""
from collections import defaultdict
import torch
from .paired_data import PairedReactionDataset
from .phase15_audit import canonical_hash


class SessionPopulation:
    def __init__(self,root_dir,split,clip_length=128):
        self.dataset=PairedReactionDataset(root_dir,split,clip_length,crop_mode='center')
        self.split,self.clip_length=split,clip_length
        self.sources=defaultdict(list);self.references=defaultdict(list);self.reference_paths={}
        source_seen=set()
        for index,path in enumerate(self.dataset.records):
            resolved=path.resolve()
            if not resolved.is_relative_to(self.dataset.directory.resolve()):
                raise ValueError('source escapes split')
            if resolved not in source_seen:
                self.sources[path.parent.name].append(index);source_seen.add(resolved)
        facial=self.dataset.directory/'facial-attributes'
        seen=set()
        for path in sorted((facial/'listener').glob('*/*.npy')):
            resolved=path.resolve()
            if not resolved.is_relative_to((facial/'listener').resolve()):
                raise ValueError('reference escapes split/listener directory')
            if resolved in seen:continue
            seen.add(resolved)
            relative=resolved.relative_to(facial.resolve())
            ref_id=relative.with_suffix('').as_posix()
            session=relative.parts[1]
            self.references[session].append(ref_id)
            self.reference_paths[ref_id]=facial/relative
        for session in self.sources:
            if not self.references[session]:raise ValueError(f'no references for {split}/{session}')

    def load_reference(self,ref_id):
        if ref_id not in self.reference_paths:raise KeyError('reference is outside this split pool')
        value=self.dataset._load(self.reference_paths[ref_id])
        if value.ndim!=2 or value.shape[1]!=25 or len(value)<1:raise ValueError('invalid listener reference')
        start=max(0,len(value)-self.clip_length)//2
        length=min(self.clip_length,len(value)-start)
        reaction=value.new_zeros(self.clip_length,25);reaction[:length]=value[start:start+length]
        return dict(reaction=reaction,length=torch.tensor(length),record_id=ref_id,
                    session_id=ref_id.split('/')[1],crop_start=start,total_length=len(value))

    def audit(self):
        return dict(split=self.split,source_population=sum(map(len,self.sources.values())),
                    reference_population=len(self.reference_paths),
                    sessions={s:dict(source_count=len(self.sources[s]),reference_count=len(self.references[s]),
                                     source_indices=self.sources[s],reference_ids=self.references[s]) for s in sorted(self.sources)})


def make_schedule(pool,steps=128,session_seed=2001,source_seed=2002,reference_seed=2003,noise_seed=789):
    sessions=sorted(pool.sources)
    weights=torch.tensor([len(pool.sources[s]) for s in sessions],dtype=torch.float64)
    session_rng=torch.Generator().manual_seed(session_seed)
    source_rng=torch.Generator().manual_seed(source_seed)
    ref_rng=torch.Generator().manual_seed(reference_seed)
    noise_rng=torch.Generator().manual_seed(noise_seed)
    records=[];noise=[]
    for step in range(steps):
        session=sessions[int(torch.multinomial(weights,1,replacement=True,generator=session_rng))]
        pool_sources=pool.sources[session]
        draws=torch.randint(len(pool_sources),(4,),generator=source_rng).tolist()
        source_indices=[pool_sources[i] for i in draws]
        refs=pool.references[session]
        ref_indices=torch.randperm(len(refs),generator=ref_rng)[:min(8,len(refs))].tolist()
        records.append(dict(step=step+1,session_id=session,source_indices=source_indices,
                            reference_ids=[refs[i] for i in ref_indices],banks=[[0,1],[2,3]]))
        noise.append(torch.randn(4,4,32,generator=noise_rng))
    noises=torch.stack(noise)
    import hashlib
    hashes=dict(session_schedule_hash=canonical_hash([r['session_id'] for r in records]),
                source_occurrence_hash=canonical_hash([r['source_indices'] for r in records]),
                reference_selection_hash=canonical_hash([r['reference_ids'] for r in records]),
                noise_hash=hashlib.sha256(noises.numpy().tobytes()).hexdigest())
    return records,noises,hashes
