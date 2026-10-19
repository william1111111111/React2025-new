"""Score frozen prediction banks against identically aligned raw all-session GT.

This is a separate representation diagnostic, not a replacement for the frozen
processed-GT evaluation protocol. Native paired GT uses source crop time;
other same-session listener GT use the existing relative-time resampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--asset-mam-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert len(args.labels)==len(args.results)==len(set(args.labels))
    torch.set_num_threads(4)
    data=PairedReactionDataset(args.data_root,'val',750,crop_mode='center',
                              normalization_dir=args.asset_mam_root/'external/FaceVerse')
    banks={label:torch.load(path.with_suffix('.predictions.pt'),map_location='cpu')
           for label,path in zip(args.labels,args.results)}
    metadata={label:json.loads(path.read_text()) for label,path in zip(args.labels,args.results)}
    base=banks[args.labels[0]]['clip_ids']
    assert len(base)==len(data)==571
    for label,bank in banks.items():
        assert bank['clip_ids']==base
        assert bank['checkpoint_sha256']==metadata[label]['checkpoint_sha256']
        assert [r['target_cache_key'] for r in metadata[label]['rows']]==[
            r['target_cache_key'] for r in metadata[args.labels[0]]['rows']]
    facial=args.data_root/'val/facial-attributes'
    paths={s:sorted((facial/'listener'/s).glob('*.npy')) for s in {k.split('/')[1] for k in base}}
    cache={}
    def load(path):
        if path not in cache:
            cache[path]=torch.from_numpy(np.load(path,allow_pickle=False).astype(np.float32))
        return cache[path]
    rows=[]
    started=time.monotonic()
    for i in range(len(data)):
        sample=data[i]
        assert sample['clip_id']==base[i]
        paired=(facial/base[i].replace('speaker/','listener/',1)).with_suffix('.npy')
        ordered=[paired,*[p for p in paths[sample['session_id']] if p!=paired]]
        targets=aligned_targets([load(path) for path in ordered],int(sample['source_lengths']),int(sample['crop_start']))
        pred=torch.cat([banks[label]['official_predictions'][i] for label in args.labels])
        scores=candidate_frc(targets.numpy(),pred.numpy()).reshape(len(args.labels),10)
        rows.append({'clip_id':base[i],'session':sample['session_id'],
                     'raw_FRC':{a:float(scores[j].sum()) for j,a in enumerate(args.labels)},
                     'per_slot':{a:scores[j].tolist() for j,a in enumerate(args.labels)}})
        if (i+1)%100==0: print(f'Raw all-session FRC {i+1}/571',flush=True)
    summary={a:{'raw_FRC':float(np.mean([r['raw_FRC'][a] for r in rows])),
                'per_slot':np.mean([r['per_slot'][a] for r in rows],axis=0).tolist(),
                'per_session':{s:float(np.mean([r['raw_FRC'][a] for r in rows if r['session']==s])) for s in paths}}
             for a in args.labels}
    result={'protocol':__doc__,'source_sha256':sha256_file(Path(__file__)),
            'input_result_sha256':{a:sha256_file(p) for a,p in zip(args.labels,args.results)},
            'checkpoint_sha256':{a:banks[a]['checkpoint_sha256'] for a in args.labels},
            'summary':summary,'rows':rows,'seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
