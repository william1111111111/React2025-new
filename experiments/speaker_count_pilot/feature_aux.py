"""Descriptive existing-feature activity during candidate voice windows; no identity gate."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/speaker_count_spectral_v1'
def main():
 rows=[]
 for r in json.loads((OUT/'SUMMARY.json').read_text())['records']:
  p=OUT/(r['id']+'.json');record=json.loads(p.read_text());rel=r['relative']
  paths={k:ROOT/'data/train'/folder/'speaker'/(rel+ext) for k,folder,ext in [('au','facial-attributes','.npy'),('exp','coefficients','.npy'),('video','video-face-crop','.mp4')]}
  j=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(paths['video'])]));t=np.array([float(x['best_effort_timestamp_time']) for x in j['frames']]);dt=np.median(np.diff(t));au=np.load(paths['au'])[:,:15];ex=np.load(paths['exp']).reshape(-1,58)[:,:52];assert len(t)==len(au)==len(ex)
  features={name:gaussian_filter1d(np.abs(np.diff(x,axis=0,prepend=x[:1])).mean(1),.08/dt) for name,x in [('anonymous_AU_switching',au),('global_expression_motion',ex)]}
  groups=[]
  for c in range(record['candidate_speaker_count'] or 0):
   mask=np.zeros(len(t),bool)
   for w in record['windows']:
    if w['candidate_voice']=='voice_'+str(c):mask|=(t>=w['start_s'])&(t<w['end_s'])
   groups.append(dict(candidate_voice='voice_'+str(c),feature_frames=int(mask.sum()),activity={name:dict(mean=float(x[mask].mean()) if mask.any() else None,relative_to_recording_mean=float(x[mask].mean()/(x.mean()+1e-12)) if mask.any() else None) for name,x in features.items()}))
  rows.append(dict(id=r['id'],groups=groups,source_identity='unknown',mapping='feature row assigned corresponding video PTS; compare to WAV times at zero offset, unverified',input_hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}))
 (OUT/'feature_aux.json').write_text(json.dumps(dict(note='Descriptive only. AU mapping unresolved; no mouth-specific claim. Zero-offset activity cannot verify identity. No effect on estimated speaker count.',records=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
