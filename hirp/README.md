# HiRP scratch v0

Independent stochastic conditional trajectory generator. All network parameters
are newly initialized. No legacy model imports or checkpoint loading.

```python
from hirp import HiRPNet
from hirp.data_adapter import generator_inputs
model = HiRPNet()
inputs = generator_inputs(batch)
predictions = model.sample(**inputs, sample_count=10)
lengths = inputs["lengths"]
# predictions[b, :, :lengths[b]] is [10, valid_T, 25].
```

For Phase 1 use `paired_data.PairedReactionDataset` and `paired_model_inputs`.
The model receives only `source_lengths`; the paired ES mask uses `pair_lengths`.
The older compatibility adapter still maps legacy `length` and never passes `target`.
The legacy paired dataset uses min(source length, paired target length) as
`length`; inference should use source-only lengths. Existing 3DMM normalization
uses FaceVerse mean/std. No normalization is performed inside this network.
The legacy evaluation collate expects `targets` (plural), whereas paired mode
provides `target`; this package does not invoke that collate or import its
module (which imports Query-Mamba constants). Official evaluation runner and
serialization are not implemented yet.

FiLM weights use normal std=0.001 and zero biases: exact zero weights would
prevent the required nonzero prior gradient on the first backward.
Sequences must have 1 <= length <= T; empty examples are explicitly rejected.
Deterministic sampling contracts apply in eval mode with supplied noise.
`sample` temporarily enters eval/no_grad and restores module modes. GPU kernels
can introduce floating point differences across batch sizes (observed prefix
error 2.98e-7). Noise must match prior output dtype/device, including under AMP.

Paired Energy Score uses float32, fixed positive [25] channel scales, RMS
Euclidean distance with epsilon, and off-diagonal unbiased self averaging.
K=1 generation is supported, but the loss rejects it. No diagnostic biased mode.

Validation: `.venv/bin/python -m pytest hirp/tests -v -s`:
38 passed in 4.96s. Full output: tests/pytest_results.txt.
Small CUDA smoke on GPU 7 passed; results in tests/cuda_smoke_results.txt.
Reproduce with `.venv/bin/python hirp/tests/cuda_smoke.py` (uses GPU 7).

Git: with user authorization, a new repository was initialized on branch
`agent/hirp-scratch-v0`. Its root commit contains only the independent HiRP
package and validation records; there is no historical base commit. Existing
project files and experiment artifacts remain outside this initial commit.
No push or long training was run.

Phase 1 now includes an independent paired loader and bounded training runner.
See ../PHASE1_REPORT.md for real-data results and limitations.
Not implemented: official evaluation runner,
group loss/sampler, local temporal noise, VAE/KL/GMM/diffusion/flow/Mamba,
emotion query, set OT, CVaR, or additional losses.
