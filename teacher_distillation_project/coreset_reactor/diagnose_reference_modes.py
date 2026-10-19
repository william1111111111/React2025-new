"""TRAIN64 mechanism diagnostic: fixed-mode fidelity and latent reference use."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import load_teacher,sha256_file
from coreset_reactor.train_mode_reference_teacher import ModeReferenceBank
from coreset_reactor.train_latent_reference_teacher import reference_ccc


def describe(x):
    return torch.cat((x.mean(1),x.std(1,unbiased=False)),dim=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','asset-mam-root','frozen-root','convergence-root','uniform-root','mode-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    data=PairedReactionDataset(args.data_root,'train',750,crop_mode='center',normalization_dir=args.asset_mam_root/'external/FaceVerse')
    mapping=json.loads((args.frozen_root/'manifests/T1/rank_00.json').read_text())['mapping']
    refs=ModeReferenceBank(args.data_root,8)
    scale=refs.descriptor_scale.to(device)
    paths={'T1':[args.convergence_root/f'arms/T1/rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)],
           'uniform':[args.uniform_root/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)],
           'mode_prior':[args.mode_root/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]}
    models={a:[load_teacher(path,device)[0].eval() for path in pp] for a,pp in paths.items()}
    ids=sorted(np.random.default_rng(1234).choice(len(data),size=64,replace=False).tolist())
    rows=[]; started=time.monotonic()
    with torch.inference_mode():
        for index in ids:
            sample=data[index]; length=int(sample['source_lengths'])
            inputs=[sample[k][None,:length].to(device) for k in ('speaker_audio','speaker_emotion','speaker_3dmm')]
            inputs.append(torch.tensor([length],device=device))
            group=refs.lookup['listener/'+sample['session_id']]
            bank=refs.at(group,length,device)
            targets=mapping[sample['clip_id']+'.npy'][1:]
            anchor=torch.tensor([refs.target_indices[t] for t in targets],device=device)
            fixed=bank[anchor]
            log_prior=refs.prior(group,length,device)[anchor]
            desc=describe(bank)/scale
            fixed_desc=desc[anchor]
            cost=(fixed_desc[:,None]-desc[None]).square().mean(-1)
            entry={'clip_id':sample['clip_id'],'arms':{}}
            for arm,ensemble in models.items():
                prediction=torch.stack([m(*inputs)['prediction'][0] for m in ensemble])
                official=prediction.clone(); official[:,:,:15]=official[:,:,:15].round()
                ccc=reference_ccc(prediction,bank)
                if arm=='T1':
                    posterior=torch.zeros_like(ccc).scatter_(1,anchor[:,None],1.)
                else:
                    logits=ccc/.05 + (log_prior if arm=='mode_prior' else 0)
                    posterior=torch.softmax(logits,dim=-1)
                entropy=-(posterior*posterior.clamp_min(1e-30).log()).sum(-1)
                values={
                    'soft_mode_descriptor_MSE':float(((describe(prediction)/scale-fixed_desc)**2).mean()),
                    'rounded_mode_descriptor_MSE':float(((describe(official)/scale-fixed_desc)**2).mean()),
                    'fixed_trajectory_MSE':float((prediction-fixed).square().mean()),
                    'fixed_reference_CCC':float(ccc.gather(1,anchor[:,None]).mean()),
                    'best_session_CCC':float(ccc.max(-1).values.mean()),
                    'posterior_expected_mode_displacement':float((posterior*cost).sum(-1).mean()),
                    'posterior_effective_references':float(entropy.exp().mean()),
                    'posterior_fixed_GT_mass':float(posterior.gather(1,anchor[:,None]).mean()),
                    'posterior_argmax_is_fixed_GT_fraction':float((posterior.argmax(-1)==anchor).float().mean()),
                }
                entry['arms'][arm]=values
            rows.append(entry)
    summary={a:{k:float(np.mean([r['arms'][a][k] for r in rows])) for k in rows[0]['arms'][a]} for a in models}
    result={'protocol':__doc__,'split':'train','source_role':'speaker','seed':1234,'contexts':64,
            'caveat':'post-hoc TRAIN diagnostic, not VAL performance or causal evidence by itself',
            'source_sha256':sha256_file(Path(__file__)),
            'target_manifest_sha256':sha256_file(args.frozen_root/'manifests/T1/rank_00.json'),
            'checkpoint_hashes':{a:[sha256_file(p) for p in pp] for a,pp in paths.items()},
            'summary':summary,'rows':rows,'seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
