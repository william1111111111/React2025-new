"""Exact mean-mode versus temporal-residual decomposition of official FRDiv."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from coreset_reactor.teacher_cache import sha256_file


def decompose(prediction):
    p=prediction.double()
    k,_,d=p.shape
    if k<2 or d!=25: raise ValueError('Expected [K>=2,T,25]')
    factor=2*k/(k-1)
    means=p.mean(1)
    residual=p-means[:,None]
    channel=factor*p.var(0,unbiased=False).mean(0)/d
    level=float(factor*means.var(0,unbiased=False).mean())
    dynamic=float(factor*residual.var(0,unbiased=False).mean())
    total=float(channel.sum())
    np.testing.assert_allclose(level+dynamic,total,rtol=1e-12,atol=1e-12)
    return {'FRDiv':total,'time_mean_component':level,'temporal_residual_component':dynamic,
            'AU_weighted_component':float(channel[:15].sum()),
            'VA_weighted_component':float(channel[15:17].sum()),
            'expression_weighted_component':float(channel[17:].sum()),
            'mean_AU_activation_range':float((means[:,:15].max(0).values-means[:,:15].min(0).values).mean()),
            'valence_mean_range':float(means[:,15].max()-means[:,15].min()),
            'arousal_mean_range':float(means[:,16].max()-means[:,16].min()),
            'distinct_dominant_expression_means':len(set(means[:,17:].argmax(-1).tolist()))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert len(args.labels)==len(args.results)==len(set(args.labels))
    torch.set_num_threads(4)
    result={}; rows=[]; ids=None
    for label,path in zip(args.labels,args.results):
        meta=json.loads(path.read_text())
        bank=torch.load(path.with_suffix('.predictions.pt'),map_location='cpu')
        assert bank['checkpoint_sha256']==meta['checkpoint_sha256']
        if ids is None: ids=bank['clip_ids']
        assert bank['clip_ids']==ids and len(ids)==571
        measurements=[decompose(x) for x in bank['official_predictions']]
        means={k:float(np.mean([r[k] for r in measurements])) for k in measurements[0]}
        np.testing.assert_allclose(means['FRDiv'],meta['metrics']['FRDiv'],atol=1e-6,rtol=1e-5)
        result[label]={'mean':means,'input_result_sha256':sha256_file(path),
                       'checkpoint_sha256':meta['checkpoint_sha256']}
        for i,value in enumerate(measurements):
            if len(rows)<=i: rows.append({'clip_id':ids[i],'arms':{}})
            rows[i]['arms'][label]=value
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'protocol':__doc__,'source_sha256':sha256_file(Path(__file__)),
                                      'summary':result,'rows':rows},indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
