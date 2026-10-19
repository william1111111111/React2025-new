from pathlib import Path
import pytest
import torch

from coreset_reactor.build_matched_real_gt_cache import real_target
from coreset_reactor.run_key_component_ablations import uniform_command, aligned
from regnn.conditional_data import crop_and_pad, relative_time_resample_reaction


def test_real_targets_use_native_anchor_and_relative_time_specialists():
    raw=torch.rand(23,25,generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(real_target(raw,0,4,10,15),crop_and_pad(raw,4,15),rtol=0,atol=0)
    expected=crop_and_pad(relative_time_resample_reaction(raw,10),0,15)
    torch.testing.assert_close(real_target(raw,3,4,10,15),expected,rtol=0,atol=0)
    assert (real_target(raw,3,4,10,15)[10:]==0).all()
    with pytest.raises(ValueError,match='shorter'):
        real_target(raw,0,20,10,15)


def test_uniform_command_changes_only_reference_prior_and_output_location():
    original=['python','-m','coreset_reactor.train_mode_reference_teacher',
              '--run-dir','old','--reference-effective-count','8.0',
              '--reference-temperature','.05','--max-train-steps','600']
    cmd=uniform_command(original,Path('/new'))
    assert cmd==['python','-m','coreset_reactor.train_latent_reference_teacher',
                 '--run-dir','/new','--reference-temperature','.05','--max-train-steps','600']
    assert '--reference-effective-count' in original


def test_evaluation_alignment_rejects_subsets_and_changed_gt_cache():
    baseline={'metrics':{'contexts':571},'rows':[{'clip_id':str(i),'target_cache_key':str(i)} for i in range(571)]}
    aligned(baseline,baseline)
    with pytest.raises(ValueError):aligned({'metrics':{'contexts':64},'rows':[]},baseline)
    rows=[dict(r) for r in baseline['rows']];rows[0]['target_cache_key']='different'
    with pytest.raises(ValueError):aligned({'metrics':{'contexts':571},'rows':rows},baseline)
