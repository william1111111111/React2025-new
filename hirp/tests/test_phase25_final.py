import torch
from hirp import HiRPConfig
from hirp.phase22 import make_model
from hirp.evaluate_phase25_final import SourceSerialSampler


def test_source_serial_preserves_noise_and_source_only_sampling():
    model=make_model(123,HiRPConfig(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dilations=(1,2))).eval()
    inputs=dict(speaker_audio=torch.randn(4,16,768),speaker_emotion=torch.randn(4,16,25),speaker_3dmm=torch.randn(4,16,58),lengths=torch.tensor([16,13,11,9]),sample_count=4,noise=torch.randn(4,4,32))
    full=model.sample(**inputs);serial=SourceSerialSampler(model).sample(**inputs)
    torch.testing.assert_close(full,serial,atol=2e-6,rtol=2e-5)
    assert torch.equal(serial[3,:,9:],torch.zeros_like(serial[3,:,9:]))
