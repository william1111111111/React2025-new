"""Interleaved, warmed, identical-input latency comparison of two teacher banks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import load_teacher,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('convergence-root','candidate-root','data-root','asset-mam-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    old=[args.convergence_root/f'arms/T1/rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(10)]
    new=[old[0]]+[args.candidate_root/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
    banks={label:[load_teacher(path,device)[0].eval() for path in paths] for label,paths in [('T1',old),('mode_prior',new)]}
    data=PairedReactionDataset(args.data_root,'val',750,crop_mode='center',normalization_dir=args.asset_mam_root/'external/FaceVerse')
    rows=[]
    with torch.inference_mode():
        for index in (0,140,285,430):
            sample=data[index]; length=int(sample['source_lengths'])
            inputs=[sample[k][None,:length].to(device) for k in ('speaker_audio','speaker_emotion','speaker_3dmm')]
            inputs.append(torch.tensor([length],device=device))
            def run(label):
                return torch.stack([m(*inputs)['prediction'][0] for m in banks[label]]).cpu()
            for label in banks:
                for _ in range(3): run(label)
            torch.cuda.synchronize(device)
            for repeat in range(12):
                order=('T1','mode_prior') if repeat%2==0 else ('mode_prior','T1')
                for label in order:
                    torch.cuda.synchronize(device); tick=time.perf_counter()
                    value=run(label)
                    torch.cuda.synchronize(device); elapsed=time.perf_counter()-tick
                    assert value.shape==(10,length,25)
                    rows.append({'clip_id':sample['clip_id'],'length':length,'arm':label,'repeat':repeat,'seconds':elapsed})
    result={'protocol':__doc__,'source_sha256':sha256_file(Path(__file__)),
            'device':torch.cuda.get_device_name(device),'dtype':'float32; no autocast',
            'tf32_allowed':torch.backends.cuda.matmul.allow_tf32,
            'includes':'10 sequential model forwards, stack and device-to-host copy; source I/O excluded',
            'selector_overhead_seconds':0,'warmup':3,'repetitions_per_context':12,'contexts':4,
            'checkpoint_hashes':{a:[sha256_file(p) for p in paths] for a,paths in [('T1',old),('mode_prior',new)]},
            'metrics':{a:{'mean_ms':1000*float(np.mean([r['seconds'] for r in rows if r['arm']==a])),
                          'median_ms':1000*float(np.median([r['seconds'] for r in rows if r['arm']==a])),
                          'parameters':sum(p.numel() for m in banks[a] for p in m.parameters())} for a in banks},
            'caveat':'one shared server; no exclusive GPU lock. Interleaved paired timing reduces clock/order bias.',
            'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics'],indent=2),flush=True)


if __name__=='__main__': main()
