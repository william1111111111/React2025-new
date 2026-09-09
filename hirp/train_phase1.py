"""Bounded paired-only training preparation; no official evaluation or extra loss.

python -m hirp.train_phase1 --phase tiny --output runs/phase1/tiny --device cuda:1
python -m hirp.train_phase1 --phase pilot --output runs/phase1/pilot \
    --tiny-summary runs/phase1/tiny/summary.json --device cuda:1
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from . import HiRPNet
from .paired_data import PairedReactionDataset, paired_model_inputs, pair_valid_mask
from .phase1_diagnostics import prior_mode, gradient_norm, prior_statistics, evaluate_diagnostics
from .losses import paired_energy_score


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def take_batch(batch, indices):
    return {key: value[indices] if torch.is_tensor(value) else [value[i] for i in indices.tolist()]
            for key, value in batch.items()}


def load_subset(root, split, indices):
    dataset = PairedReactionDataset(root / 'data', split, clip_length=128, crop_mode='center')
    return next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)))


def serializable_batch(batch):
    return {key: value.tolist() if torch.is_tensor(value) and value.ndim == 1 else
            {'shape': list(value.shape), 'min': value.min().item(), 'max': value.max().item()}
            if torch.is_tensor(value) else value for key, value in batch.items()}


def run(mode, train_cpu, val_cpu, steps, batch_size, device, output, noise_bank, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = HiRPNet().to(device)
    initial_hash = state_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    train, val = to_device(train_cpu, device), to_device(val_cpu, device)
    monitor = take_batch(train, torch.arange(min(8, len(train['clip_id'])), device=device))
    monitor_noise = noise_bank.to(device).expand(len(monitor['clip_id']), -1, -1)
    val_noise = noise_bank.to(device).expand(len(val['clip_id']), -1, -1)
    history, training = [], []
    data_rng, noise_rng = torch.Generator().manual_seed(456), torch.Generator().manual_seed(789)
    schedule_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    started = time.time()

    def interval(step):
        row = dict(step=step,
                   train=evaluate_diagnostics(model, monitor, monitor_noise, mode),
                   validation=evaluate_diagnostics(model, val, val_noise, mode))
        history.append(row)
        with (output / f'{mode}_intervals.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(mode=mode, step=step, train_loss=row['train']['loss'],
                              val_loss=row['validation']['loss'], self=row['validation']['self_distance'],
                              sigma=row['validation']['sigma_mean']), allow_nan=False), flush=True)

    interval(0)
    for step in range(1, steps+1):
        indices = torch.randperm(len(train_cpu['clip_id']), generator=data_rng)[:batch_size]
        batch = take_batch(train, indices.to(device))
        noise_cpu = torch.randn(batch_size, 4, 32, generator=noise_rng)
        schedule_hash.update(indices.numpy().tobytes())
        noise_hash.update(noise_cpu.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with prior_mode(model, mode):
            aux = model(**paired_model_inputs(batch), sample_count=4,
                        noise=noise_cpu.to(device), return_aux=True)
        details = paired_energy_score(aux['predictions'], batch['paired_target'],
                                      pair_valid_mask(batch), return_details=True)
        details['loss'].backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('nonfinite gradient')
        row = dict(step=step, **{key: value.item() for key, value in details.items()},
                   prior_mu_gradient_norm=gradient_norm(model.prior.mu),
                   prior_log_sigma_gradient_norm=gradient_norm(model.prior.log_sigma),
                   **prior_statistics(aux['latent_mu'].detach(), aux['latent_log_sigma'].detach()))
        if mode == 'A1' and (row['prior_mu_gradient_norm'] <= 0 or row['prior_log_sigma_gradient_norm'] <= 0):
            raise RuntimeError('conditional prior gradient died')
        optimizer.step()
        training.append(row)
        with (output / f'{mode}_training.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        if step % 10 == 0 or step == steps:
            interval(step)
    result = dict(mode=mode, steps=steps, parameter_count=parameter_count,
                  initial_state_sha256=initial_hash, schedule_sha256=schedule_hash.hexdigest(),
                  training_noise_sha256=noise_hash.hexdigest(), elapsed_seconds=time.time()-started,
                  initial=history[0], final=history[-1],
                  gradients_first=training[0], gradients_last=training[-1])
    del optimizer, model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['tiny', 'pilot'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--tiny-summary', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = Path(__file__).resolve().parents[1]
    if args.phase == 'pilot':
        if args.tiny_summary is None or not json.loads(args.tiny_summary.read_text()).get('tiny_gate_passed'):
            raise RuntimeError('pilot requires a completed, passed tiny overfit summary')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == 'tiny':
        indices, steps, batch_size = list(range(8)), 80, 8
    else:
        dataset = PairedReactionDataset(root / 'data', 'train')
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(2026))[:32].tolist()
        steps, batch_size = 64, 4
    train = load_subset(root, 'train', indices)
    val = load_subset(root, 'val', list(range(8)))
    noise_bank = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(321))
    np.save(args.output / 'validation_noise.npy', noise_bank.numpy())
    manifest = dict(phase=args.phase, device=args.device, steps=steps, batch_size=batch_size,
                    T=128, K_train=4, K_validation=4, seed=123, data_seed=456, noise_seed=789,
                    validation_noise_seed=321, optimizer='AdamW', lr=3e-4, weight_decay=.01,
                    train_indices=indices, val_indices=list(range(8)), crop_mode='center',
                    train=serializable_batch(train), validation=serializable_batch(val),
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                    git_diff=subprocess.check_output(['git','diff','--stat'], cwd=root, text=True),
                    torch_version=torch.__version__, dataset_root=str((root/'data').resolve()),
                    note='Only speaker -> paired listener. Validation first 8 val records; mechanism probe, not official metrics.')
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root/'hirp').rglob('*.py'))}
    manifest['source_sha256'] = source_hashes
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    modes = ['A1'] if args.phase == 'tiny' else ['A0','A1']
    results = {mode: run(mode, train, val, steps, batch_size, args.device, args.output, noise_bank) for mode in modes}
    summary = dict(phase=args.phase, results=results)
    if args.phase == 'tiny':
        a, b = results['A1']['initial']['train'], results['A1']['final']['train']
        # Predeclared engineering gate, not a research quality threshold.
        summary['tiny_gate_criteria'] = 'fixed-noise train ES drops >=5%; prediction std >1e-5; all A1 prior gradients finite and >0'
        summary['tiny_gate_passed'] = bool(b['loss'] < .95*a['loss'] and
            b['mean_absolute_prediction_std_across_samples'] > 1e-5)
    else:
        for key in ('initial_state_sha256','schedule_sha256','training_noise_sha256','parameter_count'):
            assert results['A0'][key] == results['A1'][key], key
        summary['matched_controls_verified'] = True
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('SUMMARY', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
