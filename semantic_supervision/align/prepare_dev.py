"""Freeze a TRAIN-only local event typer and prepare legal DEV teacher inputs."""
import json,hashlib,re,pickle,subprocess
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/semantic_controlled_v1';OLD=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    source=ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1/review_candidates.jsonl';data=[json.loads(x) for x in source.read_text().splitlines()]
    texts=[r['transcript_evidence'] for r in data];labels=[r['event_type'] for r in data]
    model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=20000,sublinear_tf=True),LogisticRegression(C=1.,class_weight='balanced',max_iter=300,random_state=123))
    model.fit(texts,labels)
    weights=OUT/'checkpoints/event_typer.pkl'
    with weights.open('wb') as f:pickle.dump(model,f)
    manifest=json.loads((ROOT/'runs/phase24/evaluation_v1/multitarget_development_manifest.json').read_text());rows=[]
    for source in manifest['sources']:
        rel=source['clip_id'].removeprefix('speaker/');cid='rec_'+hashlib.sha256(source['clip_id'].encode()).hexdigest()[:16];path=ROOT/'data/val/text/speaker'/(rel+'.txt');text=path.read_text()
        parts=[m.group().strip() for m in re.finditer(r'[^.!?]+[.!?]*',text) if m.group().strip()]
        # Long unpunctuated spans are broken at actual word boundaries, not retimed.
        chunks=[]
        for part in parts:
            ws=list(re.finditer(r'\S+',part))
            for i in range(0,len(ws),32):chunks.append(part[ws[i].start():ws[min(i+31,len(ws)-1)].end()])
        if len(chunks)>16:chunks=[chunks[i] for i in np.linspace(0,len(chunks)-1,16,dtype=int)]
        predicted=model.predict(chunks) if chunks else []
        candidates=[{'transcript_evidence':t,'event_type':str(k),'description':'local source-only event proposal','speaker_attribution':'unknown','start_s':None,'end_s':None} for t,k in zip(chunks,predicted)]
        rows.append({'id':cid,'relative':rel,'transcript_sha256':sha(path),'candidates':candidates})
    assert len(rows)==80
    job=OUT/'dev_job';job.mkdir(exist_ok=True);dest=OUT/'dev_source';dest.mkdir(exist_ok=True)
    teachers=json.loads((OLD/'job/job.json').read_text())['teachers'];teachers['event_typer_sha256']=sha(weights)
    (job/'job.json').write_text(json.dumps({'split':'val','records':rows,'teachers':teachers},indent=2))
    typing={'fit_split':'train','fit_candidates':len(texts),'input_sha256':sha(ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1/review_candidates.jsonl'),'model_sha256':sha(weights),'model':'char TFIDF(3,5), 20k features, balanced logistic C=1 seed123','dev_labels_read':False,'scores_are_calibrated':False,'dev_proposal_policy':'source transcript punctuation/32-word spans, up to16 evenly spaced spans; actual CTC timing only','train_retyping':'same frozen classifier will retype existing accepted TRAIN evidence, preserving role/time gates','proposal_difference':'TRAIN proposal windows originate from previous teacher candidates; DEV uses fixed text spans; report this distribution difference'}
    (OUT/'event_typer.json').write_text(json.dumps(typing,indent=2))
    cmd=json.loads((OLD/'sandbox_command.json').read_text())
    data=(ROOT/'data/train').resolve();val=(ROOT/'data/val').resolve()
    cmd=[str(dest) if x==str(OLD/'source') else str(job) if x==str(OLD/'job') else x.replace(str(data),str(val)) if x.startswith(str(data)) else x for x in cmd]
    # Use only GPU7 for this bounded source-only stage.
    uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
    for i,x in enumerate(cmd):
        if x=='CUDA_VISIBLE_DEVICES':cmd[i+1]=uuid
        elif x=='/dev/nvidia5':cmd[i]='/dev/nvidia7'
    (OUT/'dev_sandbox.json').write_text(json.dumps(cmd,indent=2))
    with (OUT/'dev_alignment.log').open('a') as f:subprocess.run(cmd,env={},stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
