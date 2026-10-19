import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from coreset_reactor.train_expert_student import validate_frozen_calibration
from regnn.conditional_model import ConditionalREGNNLoss
from coreset_reactor.run_student_loss_pruning import study_layout, training_command


def test_dataset_layout_preserves_budgets_and_reuses_only_existing_cells():
    base=Path('/test')
    old=study_layout(base,'REACT2024')
    new=study_layout(base,'REACT2025')
    assert (old['contexts'],old['train_contexts'],old['updates']) == (1124,6340,15850)
    assert (new['contexts'],new['train_contexts'],new['updates']) == (571,3320,8300)
    assert list(old['arms']) == ['minimal','velocity','residual']
    assert list(new['arms']) == ['minimal','residual']
    assert new['retained_velocity'] == base/'basic'
    assert new['full'] == base/'contrast'


def test_training_command_remaps_visible_gpu_and_freezes_only_residual():
    template=['python','-m','train','--run-dir','old','--device','cuda:5',
              '--relation-grad-ratio','0.25','--seed','1']
    minimal=training_command(template,Path('/minimal'),0.)
    residual=training_command(template,Path('/residual'),0.,Path('/frozen.json'))
    assert minimal[minimal.index('--device')+1]=='cuda:0'
    assert minimal[minimal.index('--relation-grad-ratio')+1]=='0'
    assert '--frozen-relation-provenance' not in minimal
    assert residual[-2:]==['--frozen-relation-provenance','/frozen.json']
    assert template[template.index('--device')+1]=='cuda:5'
    assert template[template.index('--relation-grad-ratio')+1]=='0.25'


def provenance():
    return {'cache_manifest_hash':'cache','initialization_checkpoint_hash':'init',
            'initial_student_state_sha256':'state','relation_calibration':{
                'objective':'centered_residual','scale':[.1,.2,.3],'weight':.21}}


def test_frozen_calibration_accepts_only_matched_training_inputs():
    assert validate_frozen_calibration(provenance(),'cache','init','state','centered_residual') == ([.1,.2,.3],.21)
    for key in ('cache_manifest_hash','initialization_checkpoint_hash','initial_student_state_sha256'):
        bad=copy.deepcopy(provenance());bad[key]='different'
        with pytest.raises(ValueError,match='mismatch'):
            validate_frozen_calibration(bad,'cache','init','state','centered_residual')


@pytest.mark.parametrize('field,value', [('scale',[0,1,1]),('scale',[1,2]),('weight',float('nan')),('objective','distance')])
def test_invalid_frozen_parameters_rejected(field,value):
    bad=provenance();bad['relation_calibration'][field]=value
    with pytest.raises(ValueError):
        validate_frozen_calibration(bad,'cache','init','state','centered_residual')


def test_zero_velocity_is_exact_minimal_loss_and_default_unchanged():
    g=torch.Generator().manual_seed(4)
    p=torch.rand(2,8,25,generator=g,requires_grad=True)
    t=torch.rand(2,8,25,generator=g)
    mask=torch.arange(8)[None] < torch.tensor([8,5])[:,None]
    minimal=ConditionalREGNNLoss(velocity_weight=0)(p,t,mask)
    full=ConditionalREGNNLoss()(p,t,mask)
    explicit=ConditionalREGNNLoss(velocity_weight=.05)(p,t,mask)
    torch.testing.assert_close(minimal['loss'],minimal['mse']+minimal['ccc_loss'],rtol=0,atol=0)
    torch.testing.assert_close(full['loss'],explicit['loss'],rtol=0,atol=0)
    torch.testing.assert_close(full['loss']-minimal['loss'],.05*full['velocity'],atol=1e-7,rtol=1e-5)
    grad=torch.autograd.grad(minimal['loss'],p,retain_graph=True)[0]
    direct=torch.autograd.grad(minimal['mse']+minimal['ccc_loss'],p)[0]
    torch.testing.assert_close(grad,direct,rtol=0,atol=0)
