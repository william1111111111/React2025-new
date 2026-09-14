"""Pinned pretrained BERT: dynamic padded contextual content-token means."""
import time
from pathlib import Path
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModel
from .common import OUT,sha,read,write,text_key
from .vendor.bert_semantic_experiment.reference_ops import masked_content_mean
@torch.inference_mode()
def encode(texts,tokenizer,model,device,batch_size=32):
 assert not model.training and not any(p.requires_grad for p in model.parameters())
 windows=[];counts=[];nwindows=[]
 for i,text in enumerate(texts):
  ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'] if text.strip() else [];ids=[t for t in ids if t not in (tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id)];counts.append(len(ids));nwindows.append((len(ids)+125)//126)
  for start in range(0,len(ids),126):
   chunk=ids[start:start+126];inp=tokenizer.build_inputs_with_special_tokens(chunk);mask=[int(token in (tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id)) for token in inp];windows.append((i,inp,mask,len(chunk)))
 accum=np.zeros((len(texts),768),dtype=np.float64)
 for start in range(0,len(windows),batch_size):
  rows=windows[start:start+batch_size];width=max(len(r[1]) for r in rows);ids=torch.full((len(rows),width),tokenizer.pad_token_id,dtype=torch.long,device=device);attn=torch.zeros_like(ids);special=torch.ones_like(ids)
  for k,(_,inp,mask,_) in enumerate(rows):ids[k,:len(inp)]=torch.tensor(inp,device=device);attn[k,:len(inp)]=1;special[k,:len(inp)]=torch.tensor(mask,device=device)
  hidden=model(input_ids=ids,attention_mask=attn).last_hidden_state;vec=masked_content_mean(hidden,attn,special).float().cpu().numpy()
  for (i,_,_,n),v in zip(rows,vec):accum[i]+=v*n
 return np.stack([v/max(n,1) for v,n in zip(accum,counts)]).astype(np.float32),counts,nwindows

def main():
 torch.set_num_threads(4);device='cuda:0' if torch.cuda.is_available() else 'cpu';root=Path.home()/'.cache/react2025/bert_semantic';hub=read(root/'hub_metadata.json');revision=hub['sha'];folder=root/revision
 assert len(revision)==40 and (folder/'model.safetensors').exists()
 tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False);model=AutoModel.from_pretrained(folder,local_files_only=True,trust_remote_code=False,use_safetensors=True,add_pooling_layer=False).to(device).eval()
 model.requires_grad_(False);assert model.config.hidden_size==768 and model.config.num_hidden_layers==12 and model.config.num_attention_heads==12
 source=read(OUT/'legal_texts.json');texts=sorted(set(source['train']+source['val']));tick=time.time();vec,counts,nwindows=encode(texts,tokenizer,model,device);assert np.isfinite(vec).all()
 # Rebatching replay tolerance: deterministic dropout-free pretrained encoder.
 replay,_,_=encode(texts[:8],tokenizer,model,device,batch_size=1);error=float(np.max(np.abs(replay-vec[:8])));assert error<1e-5
 probes=['the dog chased the cat','the cat chased the dog','', 'word '*300];pv,pc,pw=encode(probes,tokenizer,model,device);assert pc[2]==0 and pw[3]>1
 dest=OUT/'content';dest.mkdir(exist_ok=False);np.save(dest/'vectors.npy',vec)
 meta=dict(model_id='google-bert/bert-base-uncased',revision=revision,tokenizer_revision=revision,model_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},transformers_version=transformers.__version__,torch_version=torch.__version__,representation='last_hidden_state content-token mean, special/PAD excluded, FP32',max_window_tokens=128,content_window_tokens=126,window_pooling='content-token-count-weighted mean over nonoverlapping windows',text_to_row={text_key(t):(i if n else None) for i,(t,n) in enumerate(zip(texts,counts))},content_token_counts=counts,window_counts=nwindows,long_texts=sum(n>126 for n in counts),texts=len(texts),train_texts=len(source['train']),val_texts=len(source['val']),vectors_sha256=sha(dest/'vectors.npy'),legal_texts_sha256=sha(OUT/'legal_texts.json'),frozen_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,eval_mode=True,pooler_used=False,seconds=time.time()-tick,device=device,cache_replay_max_abs=error,order_probe_l2=float(np.linalg.norm(pv[0]-pv[1])),probe_note='different order yields contextual representation differences; not a semantic-accuracy claim',empty_probe_null=True,long_probe_windows=pw[3])
 write(dest/'index.json',meta);print({k:meta[k] for k in ['revision','texts','long_texts','seconds','cache_replay_max_abs','order_probe_l2']},flush=True)
if __name__=='__main__':main()
