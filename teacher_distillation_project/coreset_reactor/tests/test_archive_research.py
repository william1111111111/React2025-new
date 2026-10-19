import unittest
from pathlib import Path

from coreset_reactor.archive_research import checkpoint_kind, metric_records, path_strings, pointer


class ArchiveResearchTests(unittest.TestCase):
    def test_json_pointer_escaping(self):
        self.assertEqual(pointer(('a/b', '~x')), '/a~1b/~0x')

    def test_scalar_metrics_preserve_provenance(self):
        rows = list(metric_records({'arms': {'a': {'FRC': 1.1, 'contexts': 571,
                                                  'exact_FRD': None}}}))
        self.assertEqual(rows[0]['json_pointer'], '/arms/a')
        self.assertEqual(rows[0]['metrics'], {'FRC': 1.1})
        self.assertEqual(rows[0]['context'], {'contexts': 571})

    def test_deltas_not_relabelled_as_scores(self):
        rows = list(metric_records({'a_minus_b': {'FRDiv': -.1}}))
        self.assertEqual(rows[0]['record_kind'], 'delta')

    def test_no_per_context_array_promotion(self):
        self.assertEqual(list(metric_records({'rows': [{'FRC': 1.0}]})), [])

    def test_no_nan_or_boolean_promotion(self):
        self.assertEqual(list(metric_records({'FRC': True, 'FRDiv': float('nan')})), [])

    def test_weights_distinguished_from_predictions(self):
        self.assertEqual(checkpoint_kind(Path('student.pth')), 'checkpoint')
        self.assertEqual(checkpoint_kind(Path('final_checkpoint.pt')), 'checkpoint')
        self.assertEqual(checkpoint_kind(Path('optimizer.pth')), 'optimizer_state')
        self.assertEqual(checkpoint_kind(Path('shard_000.pt')), 'tensor_cache_or_prediction')

    def test_paths_are_values_not_commands(self):
        self.assertEqual(list(path_strings({'a': ['/tmp/a.pth', 'relative.pt']})), ['/tmp/a.pth'])


if __name__ == '__main__':
    unittest.main()
