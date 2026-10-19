import numpy as np
import torch

from coreset_reactor.analyze_t1_student_refinement import au_diagnostics, paired_deltas


def test_identical_predictions_have_no_au_errors():
    teacher = torch.linspace(.01, .99, 250).reshape(2, 5, 25)
    result = au_diagnostics([teacher], [teacher])
    assert result['teacher_relative_false_negative_rate'] == 0
    assert result['teacher_relative_false_positive_rate'] == 0
    assert sum(b['count'] for b in result['teacher_margin_bins']) == 150
    assert all(b['disagreement'] == 0 for b in result['teacher_margin_bins'])


def test_au_errors_are_teacher_relative_not_gt_claims():
    teacher = torch.full((2, 5, 25), .8)
    prediction = torch.full_like(teacher, .2)
    result = au_diagnostics([prediction], [teacher])
    assert result['teacher_relative_false_negative_rate'] == 1
    assert result['teacher_relative_false_positive_rate'] == 0
    assert result['teacher_margin_bins'][2]['disagreement_fraction'] == 1


def test_paired_bootstrap_constant_changes():
    reference = [dict(clip_id=f'speaker/s{i%2}/c{i}', frc=[.1]*10,
                      exact_FRD=90., FRDiv=.18) for i in range(6)]
    current = [dict(r, frc=[.101]*10, exact_FRD=89., FRDiv=.19) for r in reference]
    result = paired_deltas(current, reference)
    for metric, expected in [('FRC', .01), ('exact_FRD', -1.), ('FRDiv', .01)]:
        np.testing.assert_allclose(result[metric]['mean_delta'], expected)
        np.testing.assert_allclose(result[metric]['session_bootstrap_CI95'], [expected, expected])
        assert result[metric]['context_improvement_fraction'] == 1
