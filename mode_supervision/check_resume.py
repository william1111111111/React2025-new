import torch
from .prepare import OUT,write

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    a=torch.load(OUT/'smoke/M1-mode/latest.pt',map_location='cpu',weights_only=True);b=torch.load(OUT/'continuous3/M1-mode/latest.pt',map_location='cpu',weights_only=True)
    keys=['base','prior','inject','optimizer','prior_optimizer','rng','lambda_mode','step','warmup']
    result={k:equal(a[k],b[k]) for k in keys};assert all(result.values()),result
    write(OUT/'resume_acceptance.json',dict(equal=result,actual_decoder_steps=3,prior_warmup_steps=2,method='continuous 3 versus 2 + checkpoint reload + 1; timestamps excluded'))
    print(result)
if __name__=='__main__':main()
