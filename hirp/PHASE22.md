# Phase 2.2 public inference and resumable experiments

Historical `HiRPNet`, Phase 2/2.1 training and runs are unchanged. Use the
versioned `HiRP22` model for persistent prior semantics.

```python
from hirp.phase22 import load_checkpoint, eval_adapter
model, checkpoint = load_checkpoint('path/to/step_002000.pt', device='cuda:4')
pred = eval_adapter(model, speaker_audio=audio, speaker_emotion=emotion,
                    speaker_3dmm=normalized_face, lengths=source_lengths,
                    noise=explicit_noise)  # [B,K,T,25]
```

`model.forward`, `model.sample`, and `eval_adapter` share `prior_mode`.
`standard_normal` means `z=epsilon`; `conditional_gaussian` uses the retained
prior. `sample` temporarily enables eval/no_grad and restores module modes.
No target, reference or session ID is accepted as conditioning. Source 3DMM
must already use the recorded FaceVerse normalization. AU/VA/expression are
returned in legal output domains, with no target-based prediction selection.

For a Phase 2.1 checkpoint, pass its exact matching `training_manifest.json`
as `legacy_manifest` to the same loader. A0 is migrated to standard_normal;
unknown/missing sampling and scale metadata requires explicit overrides or
raises an error. Loader construction preserves caller RNG.

```bash
.venv/bin/python -m hirp.train_phase22 \
  --run runs/phase22/example_seed123 --device cuda:4 \
  --init-seed 123 --sampler-seed 123 --noise-seed 123 \
  --max-steps 2000 --lr 0.0001 --lambda-group 0.1 \
  --T 128 --K 4 --B 4 --eval-interval 500 --checkpoint-interval 500
```

The default arm is `all`, sequential C0/C1/C2. Every arm starts from the same
seed within a run, while different seed triples change initialization and the
sampler/noise streams. References are loaded only by C1/C2. B=4 is enforced by
the unchanged two-bank estimator. T/K/budget/intervals are configurable; the
frozen descriptor scaler remains the historical train-only T=128 scaler and
is not silently refitted when T changes. The locked reported pilot uses T=128.

Use `--arm C1 --stop-after 500` for a deliberate interruption. Resume with the
same configuration and `--arm C1 --resume path/to/checkpoint.pt`. Existing
attempt logs/checkpoints remain untouched; a new attempt restores model,
optimizer, schedule prefix, all RNG states and global step. The new attempt's
training log includes restored rows, so total row count equals actual steps.
Other configuration changes are refused. To extend a budget, choose a new run
path, larger `--max-steps`, matching remaining configuration, and explicitly
resume the old checkpoint. Full regenerated schedules preserve their prefix.
No budget extension was performed in the Phase 2.2 results.

Evaluation runs after training by replaying checkpoints at the locked eval
interval/fixed steps. It does not affect optimizer or RNG state and does not
select a best checkpoint. The experiment root must contain the frozen protocol
and evaluation plan. All runs use the strict public checkpoint loader:

```bash
.venv/bin/python -m hirp.evaluate_phase22 \
  --root runs/phase22/replicated_v1 --seed 123 --device cuda:7
```

`official_adapter` enforces K=10 and accepts only normalized source tensors,
source lengths and explicit noise. It is a sampling adapter, not a full
long-sequence stitching/official-metric runner. T=750 and the legacy target
alignment processor require separate integration validation. The repeated
80-clip development set is never called a hidden or independent test set.
