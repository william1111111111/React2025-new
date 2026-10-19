"""Load a frozen frontier recipe for inference from speaker inputs only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from coreset_reactor.teacher_cache import load_teacher, sha256_file
from coreset_reactor.teacher_quality_frontier import combine, load_banks
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.paired_data import PairedReactionDataset


class FixedTeacherEnsemble(nn.Module):
    def __init__(self, recipe, device, checkpoint_root=None):
        super().__init__()
        self.recipe = recipe
        self.models = nn.ModuleList()
        for spec in recipe['assignments']:
            path = Path(spec['checkpoint'])
            if checkpoint_root is not None:
                step = int(spec['bank'].split('_')[1])
                path = Path(checkpoint_root)/'arms'/spec['bank'].split('_')[0]/f'rank_{spec["slot"]:02d}'/f'conditional-step{step:06d}-seed1.pth'
            if sha256_file(path) != spec['sha256']:
                raise ValueError(f'Checkpoint mismatch: {path}')
            self.models.append(load_teacher(path, torch.device(device))[0])
        if len(self.models) != recipe['model_count']:
            raise ValueError('Recipe model count mismatch')
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, speaker_audio, speaker_emotion, speaker_3dmm, lengths):
        outputs = [m(speaker_audio, speaker_emotion, speaker_3dmm, lengths)['prediction'] for m in self.models]
        result = torch.stack(outputs[:10], dim=1)
        if self.recipe['kind'] == 'blend':
            alpha = self.recipe['alpha']
            result[:,1:] = (1-alpha)*result[:,1:]+alpha*torch.stack(outputs[10:], dim=1)
        rounded = result.clone()
        rounded[..., :15] = rounded[..., :15].round()
        return {'soft_prediction': result, 'prediction': rounded}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--recipe',required=True)
    p.add_argument('--convergence',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); args=p.parse_args(); torch.set_num_threads(4)
    if args.output.exists(): raise FileExistsError(args.output)
    recipes=json.loads(args.manifest.read_text()); recipe=next(r for r in recipes if r['name']==args.recipe)
    device=torch.device(args.device); torch.cuda.set_device(device)
    model=FixedTeacherEnsemble(recipe,device,args.convergence)
    banks,_=load_banks(args.convergence)
    data=PairedReactionDataset(args.data_root,'val',750,crop_mode='center',normalization_dir=args.asset_root/'external/FaceVerse')
    indices=sorted(np.random.default_rng(20260925).choice(len(data),8,replace=False).tolist())
    latencies=[]; errors=[]
    for j,index in enumerate(indices):
        item=data[index]; length=int(item['source_lengths'])
        inputs=[item[k][None,:length].to(device) for k in ('speaker_audio','speaker_emotion','speaker_3dmm')]
        lengths=torch.tensor([length],device=device)
        if j==0:
            for _ in range(3): model(*inputs,lengths)
        torch.cuda.synchronize(); start=time.perf_counter()
        actual=model(*inputs,lengths)['prediction'][0].cpu()
        torch.cuda.synchronize(); latencies.append(time.perf_counter()-start)
        expected=combine(banks[f'T1_{recipe["step"]}']['soft_predictions'][index],banks['T0_2400']['soft_predictions'][index],recipe)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        errors.append(float((actual-expected).abs().max()))
    result={'recipe':args.recipe,'contexts':len(indices),'sample_indices':indices,
            'all_cache_parity_pass':True,'max_absolute_error':max(errors),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'inference_ms_mean':float(np.mean(latencies)*1000),'inference_ms_median':float(np.median(latencies)*1000),
            'scope':'FP32 GPU 8-context inference, three warmup forwards; includes output CPU copy, excludes data loading; no GT enters forward'}
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
