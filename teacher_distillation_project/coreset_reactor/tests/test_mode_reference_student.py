import pytest

from coreset_reactor.run_mode_reference_student import validate_result_alignment


def ledger():
    return {'rows': [{'clip_id': str(i), 'target_cache_key': f'gt{i}'} for i in range(571)]}


def test_identical_full_val_protocol_passes():
    validate_result_alignment(ledger(), ledger())


@pytest.mark.parametrize('key', ['clip_id', 'target_cache_key'])
def test_reject_changed_context_or_gt_pool(key):
    result = ledger()
    result['rows'][3][key] = 'changed'
    with pytest.raises(ValueError, match='mismatch'):
        validate_result_alignment(result, ledger())


def test_reject_partial_val():
    result = {'rows': ledger()['rows'][:64]}
    with pytest.raises(ValueError, match='full VAL571'):
        validate_result_alignment(result, result)


@pytest.mark.parametrize('allowed', [None, ['train_session_A']])
def test_cache_input_reconstruction_preserves_optional_train_filter(tmp_path, monkeypatch, allowed):
    import json
    import torch
    from coreset_reactor import train_expert_student as training

    recorded = {}

    class FakeSource:
        def __init__(self, *args, **kwargs):
            recorded.update(kwargs)

        def set_epoch(self, epoch):
            recorded['epoch'] = epoch

    monkeypatch.setattr(training, 'ConditionalReactionDataset', FakeSource)
    torch.save([], tmp_path / 'shard.pt')
    manifest = {'clip_length':750, 'source_seed':1, 'view_epoch':0,
                'contexts':0, 'shards':['shard.pt']}
    if allowed is not None:
        manifest['session_allowlist'] = allowed
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    training.TeacherCacheDataset(tmp_path, path)
    assert recorded['session_allowlist'] == allowed
    assert recorded['crop_view_mode'] == 'random'
    assert recorded['crop_view_count'] == 1
    assert recorded['epoch'] == 0
