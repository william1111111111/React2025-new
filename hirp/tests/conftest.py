import pytest
import torch
from hirp import HiRPNet


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(2)
    torch.manual_seed(123)


@pytest.fixture
def model():
    return HiRPNet().eval()


@pytest.fixture
def inputs():
    return dict(speaker_audio=torch.randn(2, 32, 768),
                speaker_emotion=torch.randn(2, 32, 25),
                speaker_3dmm=torch.randn(2, 32, 58), lengths=torch.tensor([32, 19]))
