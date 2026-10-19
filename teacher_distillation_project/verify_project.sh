#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
[[ -x "$REACT_PYTHON" ]]
[[ -d "$REACT_DATA_DIR" ]]
[[ -f "$MAM_REACTOR_ROOT/pretrained_models/post_processor/checkpoint.pth" ]]
[[ -f "$ROOT/coreset_reactor/reports/react2024_reference_count4_student_seed1/student/student.pth" ]]
"$REACT_PYTHON" -m py_compile \
  "$ROOT/coreset_reactor/run_mode_reference_student.py" \
  "$ROOT/coreset_reactor/run_react2024_mode_distillation.py" \
  "$ROOT/coreset_reactor/train_expert_student.py" \
  "$ROOT/mam_reactor/regnn/expert_student.py"
"$REACT_PYTHON" - <<'PY'
from pathlib import Path
import torch
from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig
from coreset_reactor.fast_frc import candidate_frc

root = Path(__import__('os').environ['TEACHER_DISTILLATION_ROOT'])
checkpoint = root / 'coreset_reactor/reports/react2024_reference_count4_student_seed1/student/student.pth'
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
cfg = state['student_config']
model = ExpertConditionalREGNN(ExpertStudentConfig(
    backbone=ConditionalREGNNConfig(**cfg['backbone']),
    num_experts=cfg['num_experts'],
    expert_embedding_dim=cfg['expert_embedding_dim']))
model.load_state_dict(state['state_dict'], strict=True)
assert cfg['num_experts'] == 10
assert callable(candidate_frc)
print(f'OK teacher-distillation student={checkpoint.name} parameters={sum(p.numel() for p in model.parameters())}')
PY
echo "Teacher-distillation project verification passed."
