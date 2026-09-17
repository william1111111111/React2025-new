"""12-update CPU codec plumbing check; executor gradient check is explicitly unlabelled."""
from dataclasses import asdict
import json,time
import numpy as np
import torch
from .motion_codec import MotionCodec,CodecConfig
from .motion_generator import OracleProgramExecutor
from .reaction_program import ProgramBatch,require_observed_label
from .preparation.build import ROOT,OUT,ANN,read,write,sha

def main():
    dest=OUT/'implementation_smoke';dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(123);np.random.seed(123);start=time.time()
    records=[json.loads(x) for x in (ANN/'train/listener_reactions.jsonl').read_text().splitlines()[:2]];assets=read(ANN/'train/ASSET_RESOLVER_PRIVATE.json')
    lengths=torch.tensor([129,97]);y=torch.zeros(2,129,25);inputs=[]
    for i,r in enumerate(records):
        meta=assets[r['numeric_assets']['facial-attributes']];assert sha(meta['path'])==meta['sha256'];raw=np.load(meta['path']);y[i,:lengths[i]]=torch.from_numpy(raw[:lengths[i]].copy());inputs.append(meta)
    codec=MotionCodec();opt=torch.optim.AdamW(codec.parameters(),lr=1e-3);history=[]
    for step in range(13):
        out=codec(y,lengths);losses=codec.loss(y,lengths,out);row=dict(step=step,**{k:float(v.detach()) for k,v in losses.items()},used_codes=out['usage'])
        if step==12:history.append(row);break
        opt.zero_grad();losses['total'].backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in codec.parameters());opt.step();history.append(row)
    # Native reconstruction diagnostic; not a validation or official codec gate.
    rec=out['reconstruction'].detach();valid=codec.mask(lengths,y.shape[1]);block={name:float((rec[valid][:,a:b]-y[valid][:,a:b]).square().mean()) for name,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]}
    codec.eval().requires_grad_(False);target=out['tokens'].detach()
    program=ProgramBatch(torch.tensor([[0],[0]]),torch.tensor([[True],[True]]),torch.tensor([[[0.,64.,129.]],[[0.,48.,97.]]]),torch.tensor([[.5],[.5]]),lengths)
    executor=OracleProgramExecutor(['DIAGNOSTIC_PLACEHOLDER_NOT_AN_ACTION_LABEL']);loss=executor.loss(target,program,torch.rand(target.shape));loss.backward()
    grads={k:float(p.grad.norm()) for k,p in executor.named_parameters() if p.grad is not None}
    assert all(np.isfinite(v) for v in grads.values()) and grads['action.weight']>0
    assert not any(p.grad is not None for p in executor.parameters() if not p.requires_grad)
    # Guard proves pending real annotations cannot silently become executor targets.
    rejected=0
    for r in records:
        try:require_observed_label(r)
        except ValueError:rejected+=1
    assert rejected==2
    executor.eval();uniforms=torch.rand(2,2,3,target.shape[1],2);tokens=executor.generate(program,uniforms)
    decoded=[codec.decode_tokens(tokens[:,i],lengths,129) for i in range(2)]
    assert all(torch.isfinite(v).all() for v in decoded)
    result=dict(scope='CPU integration smoke only; two TRAIN crops, no validation/full metric or semantic claim',seed=123,codec_config=asdict(codec.config),codec_optimizer_updates=12,executor_optimizer_updates=0,executor_backward_checks=1,executor_program='explicit diagnostic placeholder, not observed label; never used for formal training',inputs=inputs,history=history,final_native_block_MSE=block,executor_CE=float(loss.detach()),executor_gradients=grads,pending_labels_rejected=rejected,generated_token_shape=list(tokens.shape),sampled_output_shape=list(decoded[0].shape),seconds=time.time()-start,full_codec_gate_passed=False,oracle_EXP_C_started=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reaction_program').glob('*.py')})
    write(dest/'RESULTS.json',result);print(json.dumps({k:result[k] for k in ['seconds','final_native_block_MSE','executor_CE','full_codec_gate_passed','oracle_EXP_C_started']},indent=2))
if __name__=='__main__':main()
