import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from hirp import HiRPNet
from hirp.losses import paired_energy_score
print('CUDA_AVAILABLE', torch.cuda.is_available())
if torch.cuda.is_available():
    torch.manual_seed(123)
    device = torch.device('cuda:7')
    model = HiRPNet().to(device).eval()
    inputs = dict(speaker_audio=torch.randn(1, 8, 768, device=device), speaker_emotion=torch.randn(1, 8, 25, device=device), speaker_3dmm=torch.randn(1, 8, 58, device=device), lengths=torch.tensor([6], device=device))
    noise = torch.randn(1, 10, 32, device=device)
    a = model.sample(**inputs, sample_count=4, noise=noise[:, :4])
    b = model.sample(**inputs, sample_count=10, noise=noise)
    print('CUDA_K_PREFIX_MAX_ERROR', (a-b[:, :4]).abs().max().item())
    torch.testing.assert_close(a, b[:, :4], atol=1e-6, rtol=1e-5)
    aux = model(**inputs, sample_count=2, return_aux=True)
    loss = paired_energy_score(aux['predictions'], torch.randn(1, 8, 25, device=device), aux['valid_mask'])
    loss.backward()
    grad = model.prior.log_sigma.weight.grad
    assert torch.isfinite(grad).all() and grad.norm() > 0
    print('CUDA_SHAPE', tuple(aux['predictions'].shape), 'LOSS', loss.item(), 'LOG_SIGMA_GRAD', grad.norm().item())
    print('CUDA_SMOKE PASSED')
