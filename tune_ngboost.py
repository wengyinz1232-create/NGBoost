"""Distributed 10 x 5 NGBoost search with exactly 150 boosting rounds.

Only inner clean/white-10 scores select parameters. Every outer speaker fold
has its own complete 50-candidate search. Historical experiments stay intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import statistics
import time
import warnings

import compare_ngboost as legacy
import numpy as np
import reproduce
from threadpoolctl import threadpool_limits

CONFIG = Path(__file__).with_name('ngboost_grid150.json')
PLAN = json.loads(CONFIG.read_text(encoding='utf-8'))
CONDITIONS = reproduce.CONDITIONS
CLASSES = np.arange(10)


def validate_plan():
    rates, depths = PLAN['learning_rates'], PLAN['max_depths']
    if (len(rates) != 10 or len(set(rates)) != 10
            or not all(0 < r <= 1 for r in rates)):
        raise ValueError('Require ten distinct learning rates in (0, 1].')
    if (len(depths) != 5 or len(set(depths)) != 5
            or not all(isinstance(d, int) and d > 0 for d in depths)):
        raise ValueError('Require five distinct positive integer tree depths.')
    if PLAN['n_estimators'] != 150:
        raise ValueError('This experiment requires exactly 150 boosting rounds.')
    if PLAN['outer_folds'] != 10 or PLAN['inner_folds'] != 3:
        raise ValueError('Preserve the 10 x 3 speaker-disjoint protocol.')


def params(rate, depth):
    return {'learning_rate': rate, 'max_depth': depth, 'n_estimators': 150}


def build_model(rate, depth, seed):
    from ngboost import NGBClassifier
    from ngboost.distns import k_categorical
    from ngboost.scores import LogScore
    from sklearn.tree import DecisionTreeRegressor
    return NGBClassifier(
        Dist=k_categorical(10), Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=depth,
            min_samples_leaf=PLAN['min_samples_leaf'], random_state=seed),
        n_estimators=150, learning_rate=rate,
        minibatch_frac=PLAN['minibatch_frac'], col_sample=PLAN['col_sample'],
        natural_gradient=PLAN['natural_gradient'], random_state=seed,
        verbose=False)


def fit_model(rate, depth, seed, x, y):
    model = build_model(rate, depth, seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=1, user_api='blas'):
            try:
                model.fit(x, y)
            except ValueError as error:
                # sklearn rejects non-finite pseudo-residuals with ValueError.
                # Do not misclassify unrelated programming or data errors.
                message = str(error).lower()
                if any(marker in message for marker in (
                        'contains nan', 'contains infinity', 'contains inf',
                        'value too large', 'must not contain infs or nans')):
                    raise FloatingPointError(str(error)) from error
                raise
            except OverflowError as error:
                raise FloatingPointError(str(error)) from error
    loss = np.asarray(model.evals_result['train']['LOGSCORE'])
    if len(model.base_models) != 150 or len(loss) != 150 or not np.isfinite(loss).all():
        raise FloatingPointError('Candidate did not complete 150 finite training rounds.')
    notes = sorted(set(f'{w.category.__name__}: {w.message}' for w in caught))
    return model, time.perf_counter() - start, notes


def probabilities(model, x):
    with threadpool_limits(limits=1, user_api='blas'):
        p = np.asarray(model.predict_proba(x))
    if (p.shape != (len(x), 10) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(1), 1)):
        raise FloatingPointError('Non-finite or invalid ten-class probabilities.')
    return p


def source_identity(work):
    meta = legacy.provenance(work)
    return {k: meta[k] for k in ('commit', 'packages', 'core_sha256', 'feature_sha256')} | {
        'search_script_sha256': reproduce.sha(Path(__file__)),
        'legacy_script_sha256': reproduce.sha(Path(legacy.__file__)),
        'config_sha256': reproduce.sha(CONFIG),
    }


def load_fold(work, fold):
    from evaluation import grouped_folds
    reproduce.check_data(work)
    with np.load(work/'data/mfcc_delta.npz', allow_pickle=False) as d:
        y, groups = d['y'], d['groups']
        variants = {c: d[c] for c in CONDITIONS}
    train, test = list(grouped_folds(groups, np.arange(len(y)), 10, PLAN['seed']))[fold-1]
    inner = list(grouped_folds(groups, train, 3, PLAN['seed']+fold))
    splits = {
        'train_people': np.unique(groups[train]).tolist(),
        'test_people': np.unique(groups[test]).tolist(),
        'inner_valid_people': [np.unique(groups[v]).tolist() for _, v in inner],
    }
    return y, groups, variants, train, test, inner, splits


def search(work, fold, rate_index):
    from evaluation import measure, training_rows, scaling, transform
    rate = PLAN['learning_rates'][rate_index]
    folder = work/'search'/f'fold_{fold:02}'
    output = folder/f'rate_{rate_index:02}.json'
    if output.exists():
        raise FileExistsError('Search shard already completed; use a fresh work directory.')
    y, _, variants, _, _, inner, splits = load_fold(work, fold)
    prepared = []
    for tr, va in inner:
        x, yy = training_rows(variants, y, tr, PLAN['mode'])
        stats = scaling(x)
        prepared.append((transform(x, stats), yy,
            {c: transform(variants[c][va], stats) for c in ('clean', 'white_10')}, y[va]))
    document = {'status': 'RUNNING', 'plan': PLAN, 'fold': fold,
        'rate_index': rate_index, 'identity': source_identity(work), 'splits': splits,
        'candidates': []}
    for depth in PLAN['max_depths']:
        row = {'params': params(rate, depth), 'inner': []}
        for j, (x, yy, valid, yv) in enumerate(prepared):
            start = time.perf_counter()
            try:
                model, seconds, notes = fit_model(rate, depth, PLAN['seed']+100*fold+j, x, yy)
                metrics = {c: measure(yv, probabilities(model, v), CLASSES) for c, v in valid.items()}
                row['inner'].append({'status': 'PASS', 'inner_fold': j+1,
                    'score': float(np.mean([v['macro_f1'] for v in metrics.values()])),
                    'metrics': metrics, 'fit_seconds': seconds, 'actual_stages': 150,
                    'warnings': notes})
                del model
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                # Numerical failure is evidence for an invalid candidate, not a
                # reason to omit it. Infrastructure/programming errors propagate.
                row['inner'].append({'status': 'INVALID', 'inner_fold': j+1,
                    'error': f'{type(error).__name__}: {error}',
                    'elapsed_seconds': time.perf_counter()-start})
            print(f'fold={fold} learning_rate={rate} depth={depth} inner={j+1}/3 '
                  f"status={row['inner'][-1]['status']}", flush=True)
        row['status'] = 'PASS' if all(r['status']=='PASS' for r in row['inner']) else 'INVALID'
        row['mean_score'] = float(np.mean([r['score'] for r in row['inner']])) if row['status']=='PASS' else None
        document['candidates'].append(row)
        reproduce.save(folder/f'rate_{rate_index:02}.progress.json', document)
    document['status'] = 'COMPLETE'
    reproduce.save(output, document)


def select_candidate(rows):
    expected = {(r, d) for r in PLAN['learning_rates'] for d in PLAN['max_depths']}
    seen, valid = set(), []
    for row in rows:
        p = row['params']; key = (p['learning_rate'], p['max_depth'])
        if key in seen or key not in expected or p != params(*key):
            raise ValueError('Duplicate, unexpected or non-150-round candidate.')
        seen.add(key)
        records = row['inner']
        if len(records)!=3 or [r['inner_fold'] for r in records]!=[1, 2, 3]:
            raise ValueError('Every candidate requires all three inner folds.')
        if any(r['status'] not in ('PASS', 'INVALID') for r in records):
            raise ValueError('Unfinished inner fold.')
        successful = all(r['status']=='PASS' for r in records)
        if not successful:
            if row['status']!='INVALID' or row['mean_score'] is not None:
                raise ValueError('Numerically invalid candidate was not excluded.')
            if any(not r.get('error') for r in records if r['status']=='INVALID'):
                raise ValueError('Invalid candidate requires an error record.')
        for r in records:
            if r['status']!='PASS':
                continue
            if r['actual_stages'] != 150:
                raise ValueError('Every fitted candidate must use 150 rounds.')
            m = r['metrics']
            if set(m) != {'clean', 'white_10'}:
                raise ValueError('Only inner clean/white-10 metrics may select parameters.')
            score = np.mean([m[c]['macro_f1'] for c in ('clean', 'white_10')])
            if not np.isfinite(score) or not 0<=score<=1 or not np.isclose(score, r['score'], rtol=0, atol=1e-12):
                raise ValueError('Invalid or inconsistent inner score.')
        if successful:
            mean = float(np.mean([r['score'] for r in records]))
            if row['status']!='PASS' or not np.isclose(mean, row['mean_score'], rtol=0, atol=1e-12):
                raise ValueError('Inconsistent mean validation score.')
            valid.append(row)
    if seen != expected:
        raise ValueError('Require all 50 parameter combinations; missing candidates cannot be dropped.')
    if not valid:
        raise ValueError('All 50 candidates are numerically invalid; no model can be selected.')
    return min(valid, key=lambda r: (-r['mean_score'], r['params']['max_depth'], r['params']['learning_rate']))


def load_search(work, fold, identity, splits):
    rows, hashes = [], {}
    for index, rate in enumerate(PLAN['learning_rates']):
        path = work/'search'/f'fold_{fold:02}'/f'rate_{index:02}.json'
        d = json.loads(path.read_text(encoding='utf-8'))
        if (d['status']!='COMPLETE' or d['plan']!=PLAN or d['fold']!=fold
                or d['rate_index']!=index or d['identity']!=identity or d['splits']!=splits):
            raise ValueError('Search shard protocol, data, source or speaker split mismatch.')
        if len(d['candidates'])!=5 or any(r['params']['learning_rate']!=rate for r in d['candidates']):
            raise ValueError('A learning-rate shard must contain its five depths.')
        rows.extend(d['candidates'])
        hashes[str(path.relative_to(work))] = reproduce.sha(path)
    return select_candidate(rows), rows, hashes


def fit_fold(work, fold):
    from evaluation import measure, training_rows, scaling, transform
    output = work/'folds'/f'fold_{fold:02}'
    if (output/'result.json').exists():
        raise FileExistsError('Completed fold already exists; use a fresh work directory.')
    y, groups, variants, train, test, _, splits = load_fold(work, fold)
    identity = source_identity(work)
    best, rows, hashes = load_search(work, fold, identity, splits)
    x, yy = training_rows(variants, y, train, PLAN['mode'])
    stats = scaling(x); p = best['params']
    model, seconds, notes = fit_model(p['learning_rate'], p['max_depth'], PLAN['seed']+fold, transform(x, stats), yy)
    predictions = {c: probabilities(model, transform(variants[c][test], stats)) for c in CONDITIONS}
    metrics = {c: measure(y[test], v, CLASSES) for c, v in predictions.items()}
    output.mkdir(parents=True, exist_ok=True)
    model_path = output/'ngboost.pkl'
    with model_path.open('wb') as stream:
        pickle.dump({'estimator': model, 'scaler': stats, 'classes': CLASSES,
            'selected': p, 'train_people': splits['train_people'], 'plan': PLAN}, stream, protocol=5)
    with model_path.open('rb') as stream:
        restored = pickle.load(stream)  # Only the model just written by this process.
    difference = 0.0
    for c in CONDITIONS:
        repeated = probabilities(restored['estimator'], transform(variants[c][test], restored['scaler']))
        np.testing.assert_allclose(repeated, predictions[c], rtol=1e-5, atol=1e-6)
        difference = max(difference, float(np.max(np.abs(repeated-predictions[c]))))
    prediction_path = output/'ngboost_predictions.npz'
    np.savez_compressed(prediction_path, indices=test, y=y[test], groups=groups[test], classes=CLASSES, **predictions)
    reproduce.save(output/'result.json', {'status': 'PASS', 'plan': PLAN, 'fold': fold,
        'identity': identity, 'splits': splits, 'selected': p, 'best_inner_score': best['mean_score'],
        'candidates': rows, 'search_sha256': hashes, 'metrics': metrics,
        'actual_stages': len(model.base_models), 'final_fit_seconds': seconds,
        'training_log_scores': model.evals_result['train']['LOGSCORE'], 'warnings': notes,
        'model_sha256': reproduce.sha(model_path), 'predictions_sha256': reproduce.sha(prediction_path),
        'model_reloaded_verified': True, 'reload_max_probability_difference': difference})
    print(f'OUTER_FOLD_VERIFIED fold={fold} selected={p}', flush=True)


def summarize(work):
    from evaluation import grouped_folds, measure
    files = sorted((work/'folds').glob('fold_*/result.json'))
    if len(files)!=10:
        raise ValueError('Require all ten successful outer folds.')
    records, indices, stored = [], [], []
    for fold, path in enumerate(files, 1):
        d = json.loads(path.read_text(encoding='utf-8'))
        if d['status']!='PASS' or d['plan']!=PLAN or d['fold']!=fold or d['actual_stages']!=150:
            raise ValueError('Invalid fold protocol or round count.')
        best = select_candidate(d['candidates'])
        if best['params']!=d['selected'] or not np.isclose(best['mean_score'], d['best_inner_score'], rtol=0, atol=1e-12):
            raise ValueError('Selected parameters do not match inner validation scores.')
        if not d['model_reloaded_verified'] or d['reload_max_probability_difference']>1e-6:
            raise ValueError('Model reload was not verified.')
        if records and d['identity']!=records[0]['identity']:
            raise ValueError('Mixed source, dependencies or data across folds.')
        for name, expected in d['search_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='search':
                raise ValueError('Invalid search evidence path.')
            if reproduce.sha(work/relative)!=expected:
                raise ValueError('Search evidence hash mismatch.')
        selected, rows, hashes = load_search(work, fold, d['identity'], d['splits'])
        if rows!=d['candidates'] or hashes!=d['search_sha256'] or selected['params']!=d['selected']:
            raise ValueError('Final fold does not match complete search evidence.')
        prediction_path = path.parent/'ngboost_predictions.npz'
        if reproduce.sha(prediction_path)!=d['predictions_sha256']:
            raise ValueError('Prediction hash mismatch.')
        model_path = path.parent/'ngboost.pkl'
        if model_path.exists() and reproduce.sha(model_path)!=d['model_sha256']:
            raise ValueError('Model hash mismatch.')
        with np.load(prediction_path, allow_pickle=False) as z:
            if not np.array_equal(z['classes'], CLASSES) or len(z['y'])!=1200:
                raise ValueError('Invalid outer prediction classes or count.')
            for c in CONDITIONS:
                actual = measure(z['y'], legacy.valid_probabilities(z[c], len(z['y'])), CLASSES)
                for m in reproduce.METRICS:
                    if not np.isclose(actual[m], d['metrics'][c][m], rtol=0, atol=1e-12):
                        raise ValueError('Saved outer metrics do not match predictions.')
            indices.extend(z['indices'].tolist())
            stored.append((z['indices'].copy(), z['groups'].copy(), z['y'].copy()))
        records.append(d)
    if sorted(indices)!=list(range(12000)):
        raise ValueError('Outer predictions must cover each recording exactly once.')
    groups = np.empty(12000, dtype=int); labels = np.empty(12000, dtype=int)
    for ids, people, y in stored:
        groups[ids] = people; labels[ids] = y
    if len(np.unique(groups))!=60 or not np.array_equal(np.bincount(labels), np.repeat(1200, 10)):
        raise ValueError('Invalid speaker or class inventory.')
    for d, (train, test), saved in zip(records, grouped_folds(groups, np.arange(12000), 10, PLAN['seed']), stored):
        split = {'train_people': np.unique(groups[train]).tolist(), 'test_people': np.unique(groups[test]).tolist(),
            'inner_valid_people': [np.unique(groups[v]).tolist() for _, v in grouped_folds(groups, train, 3, PLAN['seed']+d['fold'])]}
        if d['splits']!=split or not np.array_equal(saved[0], test):
            raise ValueError('Speaker partitions differ from the declared seeded folds.')
    report = work/'reports'; report.mkdir(exist_ok=True)
    summary = []
    for c in CONDITIONS:
        row = {'condition': c}
        for m in reproduce.METRICS:
            values = [d['metrics'][c][m] for d in records]
            row[m+'_mean'] = statistics.mean(values)
            row[m+'_fold_sd'] = statistics.stdev(values)
        summary.append(row)
    counts = {key: {str(v): sum(d['selected'][key]==v for d in records) for v in values}
        for key, values in [('learning_rate', PLAN['learning_rates']), ('max_depth', PLAN['max_depths'])]}
    selections = [{'fold': d['fold'], **d['selected'], 'inner_macro_f1': d['best_inner_score'],
        'final_fit_seconds': d['final_fit_seconds']} for d in records]
    for name, rows in [('summary.csv', summary), ('selected_parameters.csv', selections)]:
        with (report/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    reproduce.save(report/'audit.json', {'status': 'PASS', 'plan': PLAN, 'folds': 10,
        'candidates_per_fold': 50, 'fixed_rounds': 150, 'selection_counts': counts,
        'invalid_candidate_count': sum(r['status']=='INVALID' for d in records for r in d['candidates']),
        'result_sha256': {str(p.relative_to(work)): reproduce.sha(p) for p in files},
        'limits': ['Previously explored corpus; not independent confirmation.',
            'Fold SD is descriptive, not an independent confidence interval.',
            'Exactly 150 rounds is a fixed budget, not proof of optimal convergence.',
            'Only NGBoost is retrained in this workflow; historical baseline timings are not paired.']})
    lines = ['# NGBoost 10 x 5 search, fixed 150 rounds', '',
        'All 50 candidates were accounted for in each outer fold; selections use inner clean/white-10 macro-F1 only.', '',
        '| Condition | Macro-F1 mean | Fold SD | Log loss mean |', '|---|---:|---:|---:|']
    lines += [f"| {r['condition']} | {r['macro_f1_mean']:.6f} | {r['macro_f1_fold_sd']:.6f} | {r['log_loss_mean']:.6f} |" for r in summary]
    lines += ['', 'See selected_parameters.csv for the ten selected parameter pairs and audit.json for validation.',
        'No midpoint outcome is forced. Existing experiments remain separate.']
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('GRID150_ALL_TEN_FOLDS_VERIFIED', flush=True)


def main():
    validate_plan()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'search', 'fit', 'summarize'])
    parser.add_argument('--work-dir', type=Path, default=Path('.ngboost-grid150'))
    parser.add_argument('--fold', type=int, choices=range(1, 11))
    parser.add_argument('--rate-index', type=int, choices=range(10))
    args = parser.parse_args()
    if args.action=='plan':
        print(json.dumps(PLAN, indent=2)); return
    if args.action in ('search', 'fit') and args.fold is None:
        parser.error('--fold is required')
    if args.action=='search' and args.rate_index is None:
        parser.error('--rate-index is required')
    work = reproduce.stage(args.work_dir)
    if args.action=='search': search(work, args.fold, args.rate_index)
    elif args.action=='fit': fit_fold(work, args.fold)
    else: summarize(work)


if __name__=='__main__':
    main()
