import torch
from torch import nn


def stem(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                         nn.GELU(), nn.Linear(hidden_dim, output_dim))


class ModalityStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.audio = stem(768, 256, 128)
        self.emotion = stem(25, 128, 64)
        self.face = stem(58, 128, 64)
        self.fusion = nn.Sequential(nn.Linear(256, d_model), nn.LayerNorm(d_model))

    def forward(self, audio, emotion, face):
        return self.fusion(torch.cat((self.audio(audio), self.emotion(emotion),
                                      self.face(face)), dim=-1))
