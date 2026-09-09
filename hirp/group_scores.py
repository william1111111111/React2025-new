"""Descriptor-space distribution connections; no generator/target coupling."""
import torch


def descriptor_distances(x,x_valid,y,y_valid,eps=1e-8):
    valid=x_valid[..., :,None,:] & y_valid[...,None,:,:]
    count=valid.sum(-1)
    if not (count>0).all():raise ValueError('descriptor pair has no jointly valid coordinates')
    delta=(x[..., :,None,:].float()-y[...,None,:,:].float()).masked_fill(~valid,0)
    return (delta.square().sum(-1)/count.float()+eps).sqrt()


def descriptor_es(pred,pred_valid,refs,ref_valid):
    k=pred.shape[-2]
    if k<2 or refs.shape[-2]<1:raise ValueError('ES requires K>=2 and at least one reference')
    cross=descriptor_distances(pred,pred_valid,refs,ref_valid).mean((-1,-2))
    distances=descriptor_distances(pred,pred_valid,pred,pred_valid)
    off_diagonal=~torch.eye(k,dtype=torch.bool,device=pred.device)
    self_distance=distances[...,off_diagonal].mean(-1)
    return dict(loss=(cross-.5*self_distance).mean(),cross=cross.mean(),self=self_distance.mean())


def b2_score(pred,valid,paired,paired_valid):
    return descriptor_es(pred,valid,paired[:,None],paired_valid[:,None])


def b3_score(pred,valid,refs,ref_valid):
    return descriptor_es(pred,valid,refs,ref_valid)


def b4_score(pred,valid,refs,ref_valid):
    if pred.shape[0]!=4:raise ValueError('B4 requires four independent source occurrences')
    a,b=pred[:2].flatten(0,1),pred[2:].flatten(0,1)
    av,bv=valid[:2].flatten(0,1),valid[2:].flatten(0,1)
    cross_a=descriptor_distances(a,av,refs,ref_valid).mean()
    cross_b=descriptor_distances(b,bv,refs,ref_valid).mean()
    cross=.5*(cross_a+cross_b)
    self_distance=descriptor_distances(a,av,b,bv).mean()
    return dict(loss=cross-.5*self_distance,cross=cross,self=self_distance)


def session_mixture_score(pred,valid,refs,ref_valid):
    """Evaluation of the fixed-source mixture under COMMON random numbers.

    Source i,k and source j,k share epsilon_k; exclude equal NOISE indices for
    all source pairs, including i!=j. This estimates independent noise draws
    for the empirical uniform mixture over the fixed validation sources.
    It is not an iid off-diagonal statistic on flattened correlated samples.
    """
    s,k,d=pred.shape
    if k<2:raise ValueError('mixture evaluation requires K>=2')
    flat,mask=pred.reshape(s*k,d),valid.reshape(s*k,d)
    cross=descriptor_distances(flat,mask,refs,ref_valid).mean()
    noise_ids=torch.arange(k,device=pred.device).repeat(s)
    independent=noise_ids[:,None]!=noise_ids[None,:]
    self_distance=descriptor_distances(flat,mask,flat,mask)[independent].mean()
    return dict(loss=cross-.5*self_distance,cross=cross,self=self_distance)


def paired_correspondence(central,target,lengths):
    valid=torch.arange(central.shape[1],device=central.device)[None]<lengths[:,None]
    result={}
    for name,channels in [('overall',slice(0,25)),('AU',slice(0,15)),('VA',slice(15,17)),('expression',slice(17,25))]:
        delta=(central[...,channels].float()-target[...,channels].float()).masked_fill(~valid[...,None],0)
        result[name]=(delta.square().sum((1,2))/(lengths*delta.shape[-1]).float()+1e-8).sqrt().mean()
    return result


def toy_hierarchy():
    support=torch.tensor([[-1.],[1.]])
    distances=descriptor_distances(support,torch.ones_like(support,dtype=torch.bool),support,torch.ones_like(support,dtype=torch.bool))
    def expected_es(probability):
        weights=torch.stack((1-probability,probability),-1)
        cross=(weights[..., :,None]*.5*distances).sum((-1,-2))
        self_distance=(weights[..., :,None]*weights[...,None,:]*distances).sum((-1,-2))
        return cross-.5*self_distance
    conditional=torch.tensor([.1,.9])
    initial_b3=expected_es(conditional).mean().item()
    initial_b4=expected_es(conditional.mean()).item()
    final={}
    for name in ('B3','B4'):
        probabilities=conditional.clone().requires_grad_()
        for _ in range(100):
            loss=expected_es(probabilities).mean() if name=='B3' else expected_es(probabilities.mean())
            grad=torch.autograd.grad(loss,probabilities)[0]
            probabilities=(probabilities-.1*grad).detach().requires_grad_()
        final[name]=dict(probabilities=probabilities.detach().tolist(),
                         conditional_gap=(probabilities[1]-probabilities[0]).item(),
                         marginal_probability=probabilities.mean().item())
    return dict(initial_probabilities=[.1,.9],initial_B3_ES=initial_b3,initial_B4_ES=initial_b4,
                homogeneous_ES=expected_es(torch.tensor(.5)).item(),final=final,
                interpretation='Analytic expected ES on {-1,+1}; B3 matches each input to q=.5, B4 matches only their marginal.')
