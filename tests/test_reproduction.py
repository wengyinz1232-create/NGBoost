import sys
import tempfile
import unittest
from pathlib import Path
import reproduce


class ReproductionTests(unittest.TestCase):
    def test_current_configuration_inventory(self):
        configs = reproduce.CONFIGS
        self.assertEqual(len(configs), 12)
        self.assertEqual(sum(c['model'] == 'lightgbm_goss_l2' for c in configs.values()), 2)
        self.assertEqual(configs['mfcc_delta__logistic__clean_matched']['grid'], [.0001, .003, .01])
        self.assertTrue(all(len(c['grid']) == 3 for c in configs.values()))

    def test_stage_excludes_history_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / 'runtime'
            old_path = list(sys.path)
            try:
                reproduce.stage(runtime)
                self.assertEqual({p.name for p in runtime.iterdir()}, set(reproduce.CORE))
                reproduce.stage(runtime)
                (runtime / 'features.py').write_text('# stale source', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Runtime source changed'):
                    reproduce.stage(runtime)
            finally:
                sys.path[:] = old_path

    def test_rejects_historical_destination(self):
        for destination in (reproduce.REPO, reproduce.SOURCE, reproduce.SOURCE / 'scratch'):
            with self.assertRaises(ValueError):
                reproduce.stage(destination)

    def test_detects_corrupted_feature_cache(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('features.py', 'prepare_data.py'):
                (root / name).write_text('test', encoding='utf-8')
            (root / 'data').mkdir()
            (root / 'data/logmel.npz').write_bytes(b'corrupt')
            inventory = {'signature': {'frontend_sha256': reproduce.sha(root / 'features.py'),
                                      'prepare_sha256': reproduce.sha(root / 'prepare_data.py')},
                         'cache_sha256': {'logmel': 'wrong'}}
            (root / 'data/inventory.json').write_text(json.dumps(inventory), encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'Feature cache hash mismatch'):
                reproduce.check_data(root)


if __name__ == '__main__':
    unittest.main()
