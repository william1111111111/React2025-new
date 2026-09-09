"""Fixed differentiable 75-D trajectory descriptors; train-only frozen scaling."""
import json
from pathlib import Path
import torch
from torch import nn
from .phase15_audit import sha256_file,canonical_hash


def reaction_descriptor(trajectory,lengths):
    """[...,T,25] -> (values [...,75], coordinate_validity [...,75]).

    Population temporal std (ddof=0). Velocity is first-difference RMS.
    Length-one velocity is invalid, not an observed zero. Vector norms have
    finite zero subgradients for constant trajectories (no sqrt(0) NaN).
    """
    if trajectory.ndim<2 or trajectory.shape[-1]!=25 or lengths.shape!=trajectory.shape[:-2]:
        raise ValueError('expected [...,T,25] and matching lengths')
    t=trajectory.shape[-2]
    if lengths.dtype not in (torch.int32,torch.int64) or ((lengths<1)|(lengths>t)).any():
        raise ValueError('descriptor lengths must be integers in [1,T]')
    with torch.autocast(device_type=trajectory.device.type,enabled=False):
        x=trajectory.float()
        valid=torch.arange(t,device=x.device)<lengths[...,None]
        x=x.masked_fill(~valid[...,None],0)
        n=lengths.float()[...,None]
        mean=x.sum(-2)/n
        centered=(x-mean[...,None,:]).masked_fill(~valid[...,None],0)
        std=torch.linalg.vector_norm(centered,dim=-2)/n.sqrt()
        dv=(x[...,1:,:]-x[...,:-1,:]).masked_fill(~valid[...,1:,None],0)
        velocity=torch.linalg.vector_norm(dv,dim=-2)/(n-1).clamp_min(1).sqrt()
        feature=torch.cat((mean,std,velocity),-1)
        mask=torch.cat((torch.ones_like(mean,dtype=torch.bool),torch.ones_like(std,dtype=torch.bool),
                        (lengths>=2)[...,None].expand_as(velocity)),-1)
        return feature.masked_fill(~mask,0),mask


class DescriptorScaler(nn.Module):
    def __init__(self,mean,std,count):
        super().__init__()
        for name,value in [('mean',mean),('std',std),('count',count)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).detach().clone())
        if self.mean.shape!=(75,) or self.std.shape!=(75,) or self.count.shape!=(75,):
            raise ValueError('scaler coordinates must be [75]')

    def forward(self,features,valid):
        valid=valid & (self.count>0)
        values=(features.float()-self.mean)/self.std.clamp_min(1e-4)
        return values.masked_fill(~valid,0),valid

    @classmethod
    def load(cls,path):
        state=json.loads(Path(path).read_text())
        if state['split']!='train':raise ValueError('scaler must be fitted on TRAIN only')
        return cls(state['mean_train'],state['std_train'],state['valid_counts'])


def fit_descriptor_scaler(pool,output):
    if pool.split!='train':raise ValueError('descriptor scaler may only fit TRAIN split')
    output=Path(output)
    if output.exists():raise FileExistsError('never overwrite fitted descriptor scaler')
    features=[];masks=[];sources=[];crops=[]
    for ref_id in sorted(pool.reference_paths):
        item=pool.load_reference(ref_id)
        feature,mask=reaction_descriptor(item['reaction'],item['length'])
        features.append(feature.double());masks.append(mask)
        path=pool.reference_paths[ref_id]
        sources.append(dict(record_id=ref_id,path=str(path),sha256=sha256_file(path)))
        crops.append(dict(record_id=ref_id,crop_start=item['crop_start'],length=int(item['length'])))
    values=torch.stack(features);valid=torch.stack(masks);count=valid.sum(0)
    mean=values.masked_fill(~valid,0).sum(0)/count.clamp_min(1)
    variance=((values-mean).square().masked_fill(~valid,0)).sum(0)/count.clamp_min(1)
    state=dict(split='train',descriptor='temporal_mean_25, population_std_25, first_difference_RMS_25',
               clip_length=pool.clip_length,crop_mode='independent_center',std_floor=1e-4,
               mean_train=mean.tolist(),std_train=variance.sqrt().tolist(),valid_counts=count.tolist(),
               source_files=sources,source_data_hash=canonical_hash(sources),reference_crops=crops)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(state,indent=2,allow_nan=False))
    return DescriptorScaler.load(output)
