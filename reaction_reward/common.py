from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/reaction_reward/v1'
DATA=ROOT/'data/train'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def seed(s):return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:8],'little')%(2**32)
