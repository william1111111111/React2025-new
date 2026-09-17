"""Typed oracle inputs. No session, source filename, target sequence or latent leakage."""
from dataclasses import dataclass
import torch

@dataclass
class ProgramBatch:
    action_ids:torch.Tensor  # [B,A]; IDs come from a separately frozen observed vocabulary
    action_mask:torch.Tensor  # simultaneous/ordered actions are allowed
    timing:torch.Tensor  # [B,A,3] onset, peak, offset in crop-relative native frames
    intensity:torch.Tensor  # [B,A]; explicitly supplied observed intensity, not inferred emotion
    lengths:torch.Tensor
    def validate(self,vocabulary_size):
        b,a=self.action_ids.shape
        if self.action_ids.dtype!=torch.long or self.action_mask.dtype!=torch.bool:raise ValueError('bad program token types')
        if self.action_mask.shape!=(b,a) or self.timing.shape!=(b,a,3) or self.intensity.shape!=(b,a) or self.lengths.shape!=(b,):raise ValueError('program shapes')
        if not self.action_mask.any(1).all() or (self.lengths<1).any():raise ValueError('UNKNOWN/missing is not maintain')
        ids=self.action_ids[self.action_mask];tm=self.timing[self.action_mask];iv=self.intensity[self.action_mask]
        if ((ids<0)|(ids>=vocabulary_size)).any() or not torch.isfinite(tm).all() or not torch.isfinite(iv).all():raise ValueError('invalid actions')
        if (tm[:,0]<0).any() or (tm[:,0]>tm[:,1]).any() or (tm[:,1]>tm[:,2]).any() or (tm[:,0]>=tm[:,2]).any():raise ValueError('invalid action clock')
        bound=self.lengths[:,None].expand(b,a)[self.action_mask]
        if (tm[:,2]>bound).any() or ((iv<0)|(iv>1)).any():raise ValueError('outside crop or intensity scale')
        return self

def require_observed_label(record):
    if record.get('split')!='train' or record.get('status')!='observed' or record.get('actions') is None:
        raise ValueError('oracle training requires real observed TRAIN programs; pending candidates are not labels')
    if not record.get('provenance') or not record.get('training_eligible'):
        raise ValueError('program evidence/eligibility not established')
    return record
