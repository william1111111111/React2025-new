"""Matched low-LR continuation of frozen contrast KD, with optional AU KD.

All arms restart AdamW identically; this is a new five-epoch fine-tuning
stage, not a resume of the ten-epoch cosine schedule. No VAL labels enter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from coreset_reactor.train_expert_student import (
    ROOT, TeacherCacheDataset, set_seed, source_commit_sha,
    ConditionalREGNNConfig, ConditionalREGNNLoss,
    ExpertConditionalREGNN, ExpertStudentConfig,
)
from coreset_reactor.relation_distillation import (
    centered_residual_loss, au_decision_loss, au_logit_contrast_loss,
)
from coreset_reactor.teacher_cache import sha256_file


def gradient_norm(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    return float(sum(g.float().square().sum() for g in gradients if g is not None).sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--initialize-student', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--objective', choices=('control', 'au_decision', 'au_logit'), required=True)
    parser.add_argument('--extra-grad-ratio', type=float, default=.25)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    if args.extra_grad_ratio < 0 or args.epochs < 1:
        raise ValueError('Invalid training budget/gradient ratio')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    data = TeacherCacheDataset(args.data_root, args.cache_manifest)
    initial = torch.load(args.initialize_student, map_location='cpu')
    if initial['cache_manifest_hash'] != data.manifest['cache_manifest_hash']:
        raise ValueError('Continuation must use the same frozen teacher cache')
    calibration = initial['relation_calibration']
    if calibration['objective'] != 'centered_residual':
        raise ValueError('Initialize from the frozen signed-residual student')
    config = initial['student_config']
    model = ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=ConditionalREGNNConfig(**config['backbone']),
        num_experts=config['num_experts'], expert_embedding_dim=config['expert_embedding_dim'],
    )).to(device)
    model.load_state_dict(initial['state_dict'], strict=True)
    state_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_digest.update(name.encode())
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    criterion = ConditionalREGNNLoss(ccc_weight=1., velocity_weight=.05)
    relation_weight = float(calibration['weight'])
    relation_scale = torch.tensor(calibration['scale'], device=device)
    extra = {'au_decision': au_decision_loss, 'au_logit': au_logit_contrast_loss}.get(args.objective)

    def losses_for(batch):
        inputs = [batch[k].to(device, non_blocking=True) for k in (
            'speaker_audio', 'speaker_emotion', 'speaker_3dmm', 'length')]
        teacher = batch['teacher_predictions'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            output = model.generate(*inputs)
            prediction, mask = output['prediction'], output['valid_mask']
            losses = criterion(prediction.flatten(0, 1), teacher.flatten(0, 1), mask.flatten(0, 1))
        residual = centered_residual_loss(prediction, teacher, mask, relation_scale)
        auxiliary = extra(prediction, teacher, mask) if extra else residual.new_zeros(())
        return losses['loss'], residual, auxiliary

    # The exact same TRAIN probes and restored RNG keep the dropout and
    # subsequent source sampling stream matched, even in the control arm.
    extra_weight, probes = 0., []
    indices = np.random.default_rng(args.seed).choice(
        len(data), size=min(32, len(data)), replace=False).tolist()[:16]
    if extra and args.extra_grad_ratio:
        parameters = tuple(model.parameters())
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
            for offset in range(0, len(indices), args.batch_size):
                batch = default_collate([data[i] for i in indices[offset:offset+args.batch_size]])
                kd, residual, auxiliary = losses_for(batch)
                primary = kd + relation_weight * residual
                primary_norm = gradient_norm(primary, parameters, True)
                extra_norm = gradient_norm(auxiliary, parameters, False)
                probes.append({'primary_gradient': primary_norm, 'extra_gradient': extra_norm})
        extra_weight = min(1., args.extra_grad_ratio * float(np.median([
            p['primary_gradient'] / max(p['extra_gradient'], 1e-12) for p in probes])))
    extra_calibration = {'objective': args.objective, 'weight': extra_weight,
                         'target_gradient_ratio': args.extra_grad_ratio,
                         'TRAIN_probe_indices': indices if extra else [], 'probes': probes}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.05*args.lr)
    source_paths = [Path(__file__), ROOT/'coreset_reactor/relation_distillation.py',
                    ROOT/'coreset_reactor/train_expert_student.py',
                    ROOT/'mam_reactor/regnn/expert_student.py',
                    ROOT/'mam_reactor/regnn/conditional_model.py']
    provenance = {'git_commit_sha': source_commit_sha(),
                  'initialization_checkpoint_hash': sha256_file(args.initialize_student),
                  'initial_student_state_sha256': state_digest.hexdigest(),
                  'cache_manifest_hash': data.manifest['cache_manifest_hash'],
                  'teacher_checkpoint_hashes': data.manifest['teacher_checkpoint_hashes'],
                  'source_hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
                  'optimizer_policy': 'identical AdamW reset, not resume',
                  'parent_updates': initial['global_step'],
                  'extra_calibration': extra_calibration}
    settings = {**vars(args), 'parameter_count': sum(p.numel() for p in model.parameters()),
                'relation_weight': relation_weight, 'extra_weight': extra_weight}
    (args.run_dir/'config.json').write_text(json.dumps(settings, default=str, indent=2))
    (args.run_dir/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(extra_calibration), flush=True)
    started, step, history = time.monotonic(), 0, []
    schedule_digest = hashlib.sha256()
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            schedule_digest.update(json.dumps(batch['sample_id'], separators=(',', ':')).encode())
            optimizer.zero_grad(set_to_none=True)
            kd, residual, auxiliary = losses_for(batch)
            loss = kd + relation_weight * residual + extra_weight * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite continuation loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            row = {'epoch': epoch+1, 'step': step, 'loss': float(loss.detach()),
                   'kd_loss': float(kd.detach()), 'residual': float(residual.detach()),
                   'extra_loss': float(auxiliary.detach()), 'gradient_norm': float(norm)}
            history.append(row)
            if step == 1 or step % 100 == 0:
                print(json.dumps(row), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()
        provenance['observed_source_schedule_sha256'] = schedule_digest.hexdigest()
        checkpoint = {'state_dict': model.state_dict(), 'student_config': config,
                      'global_step': initial['global_step']+step, 'finetune_updates': step,
                      'cache_manifest_hash': data.manifest['cache_manifest_hash'],
                      'teacher_checkpoint_hashes': data.manifest['teacher_checkpoint_hashes'],
                      'relation_calibration': calibration, 'extra_calibration': extra_calibration,
                      'provenance': provenance}
        temporary = args.run_dir/'student.pth.tmp'
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.run_dir/'student.pth')
        torch.save({**checkpoint, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None},
                   args.run_dir/'training_state.pth')
        if args.max_steps and step >= args.max_steps:
            break
    settings.update(training_seconds=time.monotonic()-started, updates=step,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    (args.run_dir/'config.json').write_text(json.dumps(settings, default=str, indent=2))
    (args.run_dir/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.run_dir/'train_metrics.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
