from argparse import Namespace

import torch

from coreset_reactor.train_expert_student import ConditionalREGNNConfig
from regnn.conditional_model import ConditionalREGNN
from regnn.train_conditional_regnn import save_checkpoint


def test_mid_epoch_snapshot_preserves_state_and_rng(tmp_path):
    torch.manual_seed(5)
    model = ConditionalREGNN(ConditionalREGNNConfig(
        hidden_dim=16, temporal_layers=1, temporal_heads=2,
        graph_dim=8, graph_heads=2, graph_layers=0, max_seq_len=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    audit = {'global_step': 600, 'epoch_index': 5,
             'completed_batches_in_epoch': 85, 'batches_per_epoch': 103,
             'scheduler_position': 'before_epoch_end_step', 'resume_supported': False}
    path = tmp_path/'step600.pth'
    save_checkpoint(path, model, optimizer, scheduler, 6, Namespace(seed=1), step_audit=audit)
    assert torch.equal(rng, torch.get_rng_state())
    saved = torch.load(path, map_location='cpu')
    assert saved['step_audit'] == audit
    assert saved['scheduler'] == scheduler.state_dict()
    assert saved['optimizer']['param_groups'] == optimizer.state_dict()['param_groups']
    for name, tensor in state.items():
        assert torch.equal(saved['state_dict'][name], tensor)
        assert torch.equal(model.state_dict()[name], tensor)


def test_legacy_epoch_checkpoint_stays_compatible(tmp_path):
    model = ConditionalREGNN(ConditionalREGNNConfig(
        hidden_dim=16, temporal_layers=1, temporal_heads=2,
        graph_dim=8, graph_heads=2, graph_layers=0, max_seq_len=8))
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    path = tmp_path/'epoch.pth'
    save_checkpoint(path, model, optimizer, scheduler, 5, Namespace(seed=1))
    saved = torch.load(path, map_location='cpu')
    assert 'step_audit' not in saved
    assert saved['epoch'] == 5
