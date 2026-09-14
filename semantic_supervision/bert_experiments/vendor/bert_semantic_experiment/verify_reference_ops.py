"""CPU checks of reference algebra/config; NOT model or annotation validation."""
from collections import Counter
from pathlib import Path
import unittest
import torch
import yaml
from reference_ops import event_time_bias, masked_content_mean


class ReferenceChecks(unittest.TestCase):
    def test_byte_bag_erases_word_order(self):
        a = b'Alice thanked Bob.'
        b = b'Bob thanked Alice.'
        self.assertNotEqual(a, b)
        self.assertEqual(Counter(a), Counter(b))

    def test_contextual_masked_pool(self):
        h = torch.tensor([[[99., 99.], [1., 3.], [3., 5.], [88., 88.], [77., 77.]]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        specials = torch.tensor([[1, 0, 0, 1, 1]])
        torch.testing.assert_close(masked_content_mean(h, attention, specials), torch.tensor([[2., 4.]]))

    def test_pool_empty_is_not_fake_evidence(self):
        with self.assertRaises(ValueError):
            masked_content_mean(torch.zeros(1, 2, 3), torch.ones(1, 2), torch.ones(1, 2))

    def test_time_bias_known_values(self):
        frames = torch.tensor([[8., 10., 13., 15., 16., 30.]], dtype=torch.float64)
        bounds = torch.tensor([[[10., 13.]]], dtype=torch.float64)
        bias = event_time_bias(frames, bounds, torch.tensor([[True]]))
        torch.testing.assert_close(bias[..., 0], torch.tensor([[-2., 0., 0., 0., -.5, -8.]], dtype=torch.float64))

    def test_unknown_time_remains_unlocalized(self):
        frames = torch.arange(10, dtype=torch.float64)[None]
        bounds = torch.tensor([[[float('nan'), float('nan')]]], dtype=torch.float64)
        bias = event_time_bias(frames, bounds, torch.tensor([[False]]))
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_clock_origin_translation_invariant(self):
        frames = torch.tensor([[10.1, 12.3, 14.5, 16.8]], dtype=torch.float64)
        bounds = torch.tensor([[[11., 13.], [12., 13.5]]], dtype=torch.float64)
        known = torch.ones(1, 2, dtype=torch.bool)
        a = event_time_bias(frames, bounds, known)
        b = event_time_bias(frames + 200., bounds + 200., known)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_bad_intervals_rejected(self):
        with self.assertRaises(ValueError):
            event_time_bias(torch.zeros(1, 3), torch.tensor([[[2., 1.]]]), torch.ones(1, 1, dtype=torch.bool))

    def test_plan_consistency(self):
        cfg = yaml.safe_load((Path(__file__).parent / 'experiment_plan.yaml').read_text())
        self.assertEqual(len(cfg['arms']), 4)
        self.assertEqual(cfg['training']['steps_per_arm'], max(cfg['training']['checkpoints']))
        self.assertTrue(cfg['text_encoder']['frozen'])
        self.assertTrue(cfg['text_encoder']['eval_mode'])
        self.assertIsNone(cfg['text_encoder']['revision'])  # must be populated locally, never invented
        self.assertEqual(sum(a['relative_time_bias'] for a in cfg['arms'].values()), 1)
        self.assertTrue(all(not a['event_type'] for a in cfg['arms'].values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
