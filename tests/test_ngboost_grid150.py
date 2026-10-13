import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import reproduce
import tune_ngboost as tuning


def candidate_rows(score=0.4):
    rows = []
    for rate in tuning.PLAN['learning_rates']:
        for depth in tuning.PLAN['max_depths']:
            rows.append({'params': tuning.params(rate, depth), 'status': 'PASS',
                'mean_score': score, 'inner': [
                    {'status': 'PASS', 'inner_fold': j, 'actual_stages': 150,
                     'score': score, 'metrics': {'clean': {'macro_f1': score},
                        'white_10': {'macro_f1': score}}} for j in (1, 2, 3)]})
    return rows


class Grid150Tests(unittest.TestCase):
    def test_only_numerical_fit_errors_become_invalid_candidates(self):
        model = Mock()
        model.fit.side_effect = ValueError('Input y contains NaN.')
        with patch.object(tuning, 'build_model', return_value=model):
            with self.assertRaises(FloatingPointError):
                tuning.fit_model(1.0, 5, 123, np.zeros((2, 1)), np.zeros(2))
            model.fit.side_effect = ValueError('Unrelated configuration error')
            with self.assertRaisesRegex(ValueError, 'Unrelated'):
                tuning.fit_model(1.0, 5, 123, np.zeros((2, 1)), np.zeros(2))

    def test_requested_grid_and_real_fit_use_exactly_150_rounds(self):
        tuning.validate_plan()
        self.assertEqual(tuning.PLAN['learning_rates'], [i/10 for i in range(1, 11)])
        self.assertEqual(tuning.PLAN['max_depths'], [1, 2, 3, 4, 5])
        rng = np.random.default_rng(71)
        x = rng.normal(size=(400, 8)); y = np.tile(np.arange(10), 40)
        for rate, depth in [(0.1, 1), (1.0, 5)]:
            model, seconds, notes = tuning.fit_model(rate, depth, 123, x, y)
            self.assertEqual(model.n_estimators, 150)
            self.assertEqual(len(model.base_models), 150)
            self.assertEqual(model.Base.max_depth, depth)
            self.assertEqual(model.learning_rate, rate)
            self.assertEqual(tuning.probabilities(model, x[:20]).shape, (20, 10))

    def test_selects_joint_winner_and_breaks_ties_deterministically(self):
        rows = candidate_rows()
        winner = next(r for r in rows if r['params']==tuning.params(0.7, 4))
        winner['mean_score'] = 0.8
        for r in winner['inner']:
            r['score'] = 0.8
            r['metrics']['clean']['macro_f1'] = 0.7
            r['metrics']['white_10']['macro_f1'] = 0.9
        self.assertEqual(tuning.select_candidate(rows)['params'], tuning.params(0.7, 4))
        self.assertEqual(tuning.select_candidate(list(reversed(candidate_rows())))['params'], tuning.params(0.1, 1))

    def test_rejects_missing_duplicate_nonfinite_and_wrong_rounds(self):
        rows = candidate_rows()
        with self.assertRaisesRegex(ValueError, 'all 50'):
            tuning.select_candidate(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tuning.select_candidate(rows+[rows[0]])
        altered = copy.deepcopy(rows); altered[0]['inner'][0]['score'] = float('nan')
        with self.assertRaises(ValueError): tuning.select_candidate(altered)
        altered = copy.deepcopy(rows); altered[0]['params']['n_estimators'] = 800
        with self.assertRaises(ValueError): tuning.select_candidate(altered)
        altered = copy.deepcopy(rows); altered[0]['inner'][0]['metrics']['pink_10'] = {'macro_f1': 1.0}
        with self.assertRaisesRegex(ValueError, 'Only inner'):
            tuning.select_candidate(altered)

    def test_numerical_failures_are_retained_and_never_selected(self):
        rows = candidate_rows()
        rows[0].update(status='INVALID', mean_score=None)
        rows[0]['inner'][0] = {'status': 'INVALID', 'inner_fold': 1, 'error': 'FloatingPointError: invalid'}
        self.assertEqual(tuning.select_candidate(rows)['params'], tuning.params(0.2, 1))
        for row in rows:
            row.update(status='INVALID', mean_score=None)
            row['inner'][0] = {'status': 'INVALID', 'inner_fold': 1, 'error': 'FloatingPointError: invalid'}
        with self.assertRaisesRegex(ValueError, 'All 50'):
            tuning.select_candidate(rows)

    def test_complete_summary_recomputes_evidence_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            work = reproduce.stage(Path(temp))
            from evaluation import grouped_folds, measure
            groups = np.repeat(np.arange(60), 200); y = np.tile(np.arange(10), 1200)
            identity = {'test': 'one-source'}
            rows = candidate_rows()
            for fold, (train, test) in enumerate(grouped_folds(groups, np.arange(12000), 10, tuning.PLAN['seed']), 1):
                splits = {'train_people': np.unique(groups[train]).tolist(),
                    'test_people': np.unique(groups[test]).tolist(),
                    'inner_valid_people': [np.unique(groups[v]).tolist() for _, v in
                        grouped_folds(groups, train, 3, tuning.PLAN['seed']+fold)]}
                for rate_index in range(10):
                    reproduce.save(work/'search'/f'fold_{fold:02}'/f'rate_{rate_index:02}.json',
                        {'status': 'COMPLETE', 'plan': tuning.PLAN, 'fold': fold,
                         'rate_index': rate_index, 'identity': identity, 'splits': splits,
                         'candidates': rows[rate_index*5:(rate_index+1)*5]})
                best, loaded, hashes = tuning.load_search(work, fold, identity, splits)
                folder = work/'folds'/f'fold_{fold:02}'; folder.mkdir(parents=True)
                p = np.full((1200, 10), 0.1)
                prediction = folder/'ngboost_predictions.npz'
                np.savez_compressed(prediction, indices=test, y=y[test], groups=groups[test],
                    classes=np.arange(10), **{c: p for c in tuning.CONDITIONS})
                reproduce.save(folder/'result.json', {'status': 'PASS', 'plan': tuning.PLAN,
                    'fold': fold, 'identity': identity, 'splits': splits, 'actual_stages': 150,
                    'candidates': loaded, 'selected': best['params'], 'best_inner_score': best['mean_score'],
                    'search_sha256': hashes, 'model_reloaded_verified': True,
                    'reload_max_probability_difference': 0, 'predictions_sha256': reproduce.sha(prediction),
                    'metrics': {c: measure(y[test], p, np.arange(10)) for c in tuning.CONDITIONS},
                    'final_fit_seconds': 1})
            tuning.summarize(work)
            audit = json.loads((work/'reports/audit.json').read_text())
            self.assertEqual(audit['selection_counts']['learning_rate']['0.1'], 10)
            self.assertEqual(audit['fixed_rounds'], 150)
            path = work/'folds/fold_01/result.json'
            original = json.loads(path.read_text())
            altered = copy.deepcopy(original); altered['selected'] = tuning.params(1.0, 5)
            reproduce.save(path, altered)
            with self.assertRaisesRegex(ValueError, 'Selected parameters'):
                tuning.summarize(work)
            altered = copy.deepcopy(original); altered['metrics']['pink_10']['macro_f1'] += 0.1
            reproduce.save(path, altered)
            with self.assertRaisesRegex(ValueError, 'Saved outer metrics'):
                tuning.summarize(work)
            reproduce.save(path, original)
            shard = work/'search/fold_01/rate_00.json'
            shard.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Search evidence hash'):
                tuning.summarize(work)

    def test_missing_outer_folds_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as temp:
            work = reproduce.stage(Path(temp))
            with self.assertRaisesRegex(ValueError, 'all ten'):
                tuning.summarize(work)


if __name__=='__main__':
    unittest.main()
