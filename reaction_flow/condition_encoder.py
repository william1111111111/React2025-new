import torch
from torch import nn
from hirp.config import HiRPConfig
from hirp.model.modality_stem import ModalityStem
from hirp.model.speaker_encoder import SpeakerEncoder
class ConditionEncoder(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or HiRPConfig();self.stems=ModalityStem(self.config.d_model);self.encoder=SpeakerEncoder(self.config);self.requires_grad_(False)
    def from_parent(self,state):
        self.stems.load_state_dict({k[len('stems.'):]:v for k,v in state.items() if k.startswith('stems.')})
        self.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')})
    @torch.no_grad()
    def forward(self,speaker_audio,speaker_emotion,speaker_3dmm,lengths):
        self.eval();b,t,_=speaker_audio.shape
        if lengths.dtype not in (torch.int32,torch.int64):raise ValueError('lengths must be integer')
        lengths=lengths.to(speaker_audio.device)
        if lengths.shape!=(b,) or ((lengths<1)|(lengths>t)).any():raise ValueError('invalid source lengths')
        valid=torch.arange(t,device=lengths.device)[None]<lengths[:,None]
        inputs=[]
        for x,width in zip((speaker_audio,speaker_emotion,speaker_3dmm),(768,25,58)):
            if x.shape!=(b,t,width):raise ValueError('bad speaker shape')
            inputs.append(x.masked_fill(~valid[...,None],0))
        h,_=self.encoder(self.stems(*inputs),valid)
        return h,valid
