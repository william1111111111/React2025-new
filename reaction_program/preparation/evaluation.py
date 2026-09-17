"""Fail-closed adapter for the new frozen protocol, not a new metric implementation."""
from pathlib import Path
import numpy as np
from .build import read,sha,OUT

def load_manifest():
    path=OUT/'evaluation/manifest.json';lock=read(OUT/'evaluation/LOCK.json')
    if sha(path)!=lock['manifest_sha256']:raise ValueError('reference manifest changed')
    m=read(path)
    if m['K']!=10 or m['R']!=10 or any(len(s['targets'])!=10 for s in m['sources']):raise ValueError('K/reference protocol mismatch')
    return m

def load_processed_targets(index,manifest=None):
    m=manifest or load_manifest();s=m['sources'][index];r=s['processed_target'];p=Path(r['path'])
    if not p.is_file():raise FileNotFoundError('Historical Processor target cache missing. Restore matching SHA or create a new shared evaluation version; never silently interpolate.')
    if sha(p)!=r['sha256']:raise ValueError('Processor target cache changed')
    y=np.load(p,allow_pickle=False)
    if y.shape!=(10,s['length'],25) or not np.isfinite(y).all():raise ValueError('bad target shape/data')
    return y

def validate_prediction(pred,index,manifest=None):
    m=manifest or load_manifest();s=m['sources'][index]
    if pred.shape!=(10,s['length'],25) or not np.isfinite(pred).all():raise ValueError('require native K10 finite full trajectory')
    # Never clamp/round or pick candidates to improve scores here.
    return pred
