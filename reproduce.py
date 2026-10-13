"""Reproduce the current twelve configurations without historical artifacts.

Training equations, folds, seeds and model implementations are copied unchanged
from AudioMNIST_完整版 into an isolated runtime directory.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parent
SOURCE = REPO / 'AudioMNIST_完整版'
CORE = ('common.py', 'features.py', 'prepare_data.py', 'evaluation.py',
        'verify.py', 'verify_saved_run.py', 'predict.py', 'rf_settings.json')
RUN_ID = 'reproduction'
CONDITIONS = ('clean', 'white_20', 'white_10', 'pink_10')
METRICS = ('macro_f1', 'accuracy', 'macro_precision', 'balanced_accuracy', 'log_loss')

def configurations():
    result = {}
    rf_settings = json.loads((SOURCE / 'rf_settings.json').read_text(encoding='utf-8'))
    for rep in ('logmel', 'mfcc_delta'):
        for model in ('logistic', 'forest', 'lightgbm_goss'):
            for mode in ('clean_matched', 'augmented'):
                actual = 'lightgbm_goss_l2' if rep == 'mfcc_delta' and model == 'lightgbm_goss' else model
                grid = {'logistic': [.0001, .1, 100.], 'forest': None,
                        'lightgbm_goss': [3, 63, 255], 'lightgbm_goss_l2': [0., .1, 1.]}[actual]
                if rep == 'mfcc_delta' and model == 'logistic':
                    grid = [.0001, .003, .01] if mode == 'clean_matched' else [.01, .03, .1]
                key = f'{rep}__{actual}__{mode}'
                if model == 'forest': grid = list(rf_settings['grids'][key])
                result[key] = {'representation': rep, 'model': actual, 'mode': mode, 'grid': grid}
    return result

CONFIGS = configurations()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def stage(work):
    work = work.resolve()
    if work in (REPO, SOURCE) or SOURCE in work.parents:
        raise ValueError('Use an isolated work directory outside the historical source directory.')
    work.mkdir(parents=True, exist_ok=True)
    for name in CORE:
        src, dest = SOURCE / name, work / name
        if dest.exists():
            if sha(src) != sha(dest):
                raise ValueError(f'Runtime source changed: {name}. Choose a NEW --work-dir.')
        else:
            shutil.copy2(src, dest)
    sys.path.insert(0, str(work))
    return work

def prepare(work, archive):
    if archive:
        archive = archive.resolve()
        target = work / 'data_raw/audiomnist.zip'
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if sha(target) != sha(archive):
                raise ValueError('Existing raw archive differs; choose a new work directory.')
        elif archive != target:
            shutil.copy2(archive, target)
    from prepare_data import prepare as prepare_data
    prepare_data(download=True)

def check_data(work):
    """Validate restored/cached features before training, without needing audio."""
    import numpy as np
    inventory = json.loads((work / 'data/inventory.json').read_text(encoding='utf-8'))
    for name, field in [('features.py', 'frontend_sha256'), ('prepare_data.py', 'prepare_sha256')]:
        if sha(work / name) != inventory['signature'][field]:
            raise ValueError(f'Cached features do not match {name}')
    identity = None
    for rep, width in [('logmel', 256), ('mfcc_delta', 320)]:
        path = work / f'data/{rep}.npz'
        assert sha(path) == inventory['cache_sha256'][rep], 'Feature cache hash mismatch'
        with np.load(path) as d:
            current = [d[k] for k in ('y', 'groups', 'ids')]
            assert len(d['y']) == len(set(d['ids'])) == 12000
            assert len(np.unique(d['groups'])) == 60
            assert np.array_equal(np.bincount(d['y']), np.repeat(1200, 10))
            assert all(d[c].shape == (12000, width) and np.isfinite(d[c]).all() for c in CONDITIONS)
            if identity is not None:
                assert all(np.array_equal(a, b) for a, b in zip(identity, current))
            identity = current
    return inventory

def result_path(work, key):
    return work / 'runs' / RUN_ID / key / 'result.json'

def train(work, keys, jobs):
    check_data(work)
    if len(keys) == 1:
        from evaluation import run_configuration
        c = CONFIGS[keys[0]]
        run_configuration(c['representation'], c['model'], c['mode'], RUN_ID, c['grid'])
        return
    def worker(key):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'train',
                        '--work-dir', str(work), '--config', key], check=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(worker, keys))

def verify(work, keys):
    """Check artifacts and recompute predictions; midpoint status is separate."""
    import pickle
    import numpy as np
    from common import SEED
    from evaluation import grouped_folds, model_for, proba, scaling, training_rows, transform
    from verify import unit_checks
    from verify_saved_run import verify as verify_saved
    unit_checks()
    check_data(work)
    results = []
    for key in keys:
        path = result_path(work, key)
        doc = json.loads(path.read_text(encoding='utf-8'))
        expected = CONFIGS[key]
        for field in ('representation', 'model', 'mode', 'grid'):
            assert doc['config'][field] == expected[field], (key, field)
        for name, value in doc['config']['signature'].items():
            target = work / 'data' / name if name.endswith('.npz') else work / name
            assert sha(target) == value, (key, name)
        audit = verify_saved(path)
        with np.load(work / f'data/{expected["representation"]}.npz') as data:
            y, groups = data['y'], data['groups']
            variants = {c: data[c] for c in CONDITIONS}
        splits = list(grouped_folds(groups, np.arange(len(y)), 10, SEED))
        checks = []
        for row in [*doc['folds'], doc['full_fit']]:
            number = row['fold']
            tr, te = splits[number-1] if number else (np.arange(len(y)), np.array([], dtype=int))
            inner = list(grouped_folds(groups, tr, 3, SEED + number))
            assert row['inner_valid_people'] == [np.unique(groups[v]).tolist() for _, v in inner]
            scores = np.asarray(row['inner_scores'])
            assert scores.shape == (3, 3) and np.isfinite(scores).all()
            assert row['selected'] == expected['grid'][int(np.argmax(scores.mean(1)))]
            mp = path.parent / 'models' / f'fold_{number:02}.pkl'
            assert sha(mp) == row['model_sha256']
            # Only this workflow's newly generated models are loaded.
            with mp.open('rb') as stream:
                bundle = pickle.load(stream)
            assert bundle['representation'] == expected['representation']
            assert bundle['mode'] == expected['mode']
            assert bundle['selected'] == row['selected']
            assert bundle['source_signature'] == sha(work / 'features.py')
            assert bundle['train_people'] == row['train_people'] == np.unique(groups[tr]).tolist()
            assert np.array_equal(bundle['classes'], np.arange(10))
            estimator = model_for(expected['model'], row['selected'], SEED + number)
            assert bundle['estimator'].get_params() == estimator.get_params()
            x, yy = training_rows(variants, y, tr, expected['mode'])
            stats = scaling(x)
            assert row['fit_rows'] == len(yy)
            assert row['train_n'] == len(tr) and row['test_n'] == len(te)
            assert all(np.array_equal(a, b) for a, b in zip(stats, bundle['scaler']))
            delta = 0.
            if number:
                with np.load(path.parent / f'predictions_{number:02}.npz') as saved:
                    for condition in CONDITIONS:
                        actual = proba(bundle['estimator'], transform(variants[condition][te], stats), bundle['classes'])
                        delta = max(delta, float(np.max(np.abs(actual - saved[condition]))))
                        assert np.isfinite(actual).all() and (actual >= 0).all()
                        assert np.allclose(actual, saved[condition], atol=1e-6, rtol=1e-5)
            else:
                assert row['metrics'] == {} and row['test_people'] == []
                p = proba(bundle['estimator'], transform(variants['clean'][:8], stats), bundle['classes'])
                assert p.shape == (8, 10) and np.allclose(p.sum(1), 1)
            checks.append({'fold': number, 'model_and_scaler_verified': True, 'max_prediction_difference': delta})
        audit['model_checks'] = checks
        audit['result_sha256'] = sha(path)
        audit['middle_strict_majority'] = audit['selected_counts'][1] > 5
        audit['warnings'] = sum(len(r['warnings']) for r in [*doc['folds'], doc['full_fit']])
        save(work / 'reports/audits' / (key + '.json'), audit)
        results.append(audit)
        print('VERIFIED', key, audit['selected_counts'], flush=True)
    return results

def summarize(work):
    import importlib.metadata
    rows, audits, documents = [], [], []
    for key in CONFIGS:
        doc = json.loads(result_path(work, key).read_text(encoding='utf-8'))
        audit = json.loads((work / 'reports/audits' / (key + '.json')).read_text(encoding='utf-8'))
        assert audit['status'] == 'PASS' and len(doc['folds']) == 10 and 'full_fit' in doc
        assert audit['result_sha256'] == sha(result_path(work, key))
        assert len(audit['model_checks']) == 11
        audits.append({'id': key, 'grid': CONFIGS[key]['grid'], 'selected_counts': audit['selected_counts'],
                       'middle_unique_mode': audit['middle_unique_mode'], 'middle_strict_majority': audit['middle_strict_majority'],
                       'warnings': audit['warnings']})
        documents.append(doc)
        for condition in CONDITIONS:
            row = {'id': key, **{k: CONFIGS[key][k] for k in ('representation', 'model', 'mode')}, 'condition': condition}
            for metric in METRICS:
                values = [r['metrics'][condition][metric] for r in doc['folds']]
                row[metric + '_mean'] = statistics.mean(values)
                row[metric + '_fold_sd'] = statistics.stdev(values)
            rows.append(row)
    reports = work / 'reports'
    with (reports / 'summary_current_12.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(reports / 'status.json', {
        'integrity_status': 'PASS', 'configurations': 12, 'outer_models': 120, 'full_data_models': 12, 'model_fits': 1320,
        'midpoint_failures': [r['id'] for r in audits if not r['middle_unique_mode']],
        'parameter_audit': audits, 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'threadpoolctl')},
        'source_sha256': {name: sha(work / name) for name in CORE},
        'result_sha256': {key: sha(result_path(work, key)) for key in CONFIGS},
        'limits': ['Engineering verification is not course-wide compliance.',
                   'RF tunes max_features with fixed depth 16, 60 trees and minimum leaf size 2.',
                   'RF grids are post-hoc and Log-Mel clean/augmented ranges differ.',
                   'MFCC logistic uses different grids by training mode.',
                   'MFCC LightGBM uses fixed seven-leaf trees and a shared L2 grid.',
                   'Data and grids were explored previously; this is a reproduction, not independent confirmation.',
                   'Error bars are fold sample SD, not independent confidence intervals.',
                   'Unique mode and strict majority are reported separately.']})
    plot(reports, rows)
    print('REPRODUCTION_COMPLETE', reports, flush=True)

def plot(reports, rows):
    os.environ.setdefault('MPLCONFIGDIR', str(reports.parent / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for i, rep in enumerate(('logmel', 'mfcc_delta')):
        for j, model in enumerate(('logistic', 'forest', 'lightgbm')):
            ax = axes[i, j]
            for mode, label, color in [('clean_matched', 'Clean training', '#b26727'), ('augmented', 'Augmented training', '#176890')]:
                values = [next(r for r in rows if r['representation'] == rep and r['model'].startswith(model) and r['mode'] == mode and r['condition'] == c) for c in CONDITIONS]
                ax.errorbar(range(4), [r['macro_f1_mean'] for r in values], yerr=[r['macro_f1_fold_sd'] for r in values], marker='o', capsize=3, label=label, color=color)
            ax.set_title(f'{rep} | {model}'); ax.set_ylim(0, 1.04)
            ax.set_xticks(range(4), ['Clean', 'White 20 dB', 'White 10 dB', 'Pink 10 dB'], rotation=18)
            ax.set_ylabel('Macro-F1 (mean +/- fold SD)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('AudioMNIST: current 12 configurations, fresh 10 x 3 speaker-disjoint CV')
    fig.supxlabel('Log-Mel RF and MFCC logistic grids differ by mode. Fold SD is not a confidence interval.', fontsize=9)
    fig.savefig(reports / 'robustness.png', dpi=160); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'self-test', 'prepare', 'train', 'verify', 'summarize', 'all'])
    parser.add_argument('--work-dir', type=Path, default=REPO / '.reproduction')
    parser.add_argument('--raw-archive', type=Path, help='Optional local fixed-version AudioMNIST ZIP; otherwise download it.')
    parser.add_argument('--config', nargs='+', choices=CONFIGS)
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    if args.action == 'list':
        print(json.dumps(CONFIGS, indent=2)); return
    work = stage(args.work_dir)
    keys = args.config or list(CONFIGS)
    if args.action == 'self-test':
        from verify import unit_checks
        unit_checks(); return
    if args.action in ('prepare', 'all'):
        prepare(work, args.raw_archive)
    if args.action in ('train', 'all'):
        train(work, keys, args.jobs)
    if args.action in ('verify', 'all'):
        verify(work, keys)
    if args.action == 'summarize' or (args.action == 'all' and set(keys) == set(CONFIGS)):
        summarize(work)

if __name__ == '__main__':
    main()
