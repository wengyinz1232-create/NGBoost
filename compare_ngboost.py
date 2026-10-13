"""Predeclared, paired MFCC+delta/white-10 augmented NGBoost comparison.

This is a new experiment, isolated from the original twelve configurations.
No outer labels or pink-noise scores select parameters or boosting stages.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import statistics as st
import subprocess
import sys
import time
import traceback
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
import reproduce

PLAN = {
    'protocol_version': 1, 'representation': 'mfcc_delta', 'mode': 'augmented',
    'outer_folds': 10, 'inner_folds': 3, 'seed': 20261009,
    'selection': 'mean(clean macro-F1, white_10 macro-F1)',
    'ngboost': {'grid': [50, 100, 200], 'parameter': 'n_estimators',
                'learning_rate': .03, 'max_depth': 3, 'min_samples_leaf': 20,
                'minibatch_frac': .5, 'col_sample': .5, 'natural_gradient': True},
    'lightgbm': {'grid': [0., .1, 1.], 'parameter': 'reg_lambda',
                 'n_estimators': 70, 'num_leaves': 7, 'learning_rate': .08,
                 'min_child_samples': 20, 'top_rate': .2, 'other_rate': .1},
    'compute': 'same runner per paired fold; BLAS=1; LightGBM threads=2; NGBoost base trees use serial fitting',
    'stage_reuse': 'Three 200-stage inner NGBoost fits score prefixes 50/100/200; selected stage count refitted on outer training data.',
    'test_policy': 'Evaluate each outer condition once after inner selection; no outer-score-based retuning.',
}
CONDITIONS = reproduce.CONDITIONS
METRICS = reproduce.METRICS
LEGACY_PLAN = copy.deepcopy(PLAN)


def configure_protocol(protocol):
    """Keep the historical grid reproducible; select the expanded grid explicitly."""
    PLAN.clear()
    PLAN.update(copy.deepcopy(LEGACY_PLAN))
    if protocol == 'extended800':
        PLAN['protocol_version'] = 2
        PLAN['ngboost']['grid'] = [200, 400, 800]
        PLAN['stage_reuse'] = ('Three 800-stage inner NGBoost fits score prefixes '
                               '200/400/800; selected stage count refitted on outer training data.')
    elif protocol != 'legacy200':
        raise ValueError(f'Unknown protocol: {protocol}')


def ng_model(stages, seed):
    from ngboost import NGBClassifier
    from ngboost.distns import k_categorical
    from ngboost.scores import LogScore
    from sklearn.tree import DecisionTreeRegressor
    p = PLAN['ngboost']
    return NGBClassifier(Dist=k_categorical(10), Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'], random_state=seed),
        n_estimators=int(stages), learning_rate=p['learning_rate'],
        minibatch_frac=p['minibatch_frac'], col_sample=p['col_sample'],
        natural_gradient=True, random_state=seed, verbose=False)


def valid_probabilities(p, n):
    p = np.asarray(p)
    if p.shape != (n, 10) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid ten-class probabilities')
    return p


def predict(model, x, name, stages=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        with threadpool_limits(limits=1, user_api='blas'):
            p = model.predict_proba(x, max_iter=stages) if name == 'ngboost' else model.predict_proba(x)
    # NGBoost's categorical outputs are p0,...,p9; LGB's class order is explicit.
    if name == 'lightgbm' and not np.array_equal(model.classes_, np.arange(10)):
        raise ValueError('Unexpected LightGBM class order')
    return valid_probabilities(p, len(x))


def fit_model(name, value, x, y, seed):
    from evaluation import model_for
    model = ng_model(value, seed) if name == 'ngboost' else model_for('lightgbm_goss_l2', value, seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=1, user_api='blas'):
            model.fit(x, y)
    elapsed = time.perf_counter() - start
    notes = sorted(set(f'{w.category.__name__}: {w.message}' for w in caught))
    if name == 'ngboost':
        loss = np.asarray(model.evals_result['train']['LOGSCORE'])
        if not np.isfinite(loss).all():
            raise FloatingPointError('Non-finite NGBoost training log score')
        if len(model.base_models) != int(value):
            raise RuntimeError(f'NGBoost stopped at {len(model.base_models)} of {value} stages; report failure rather than silently change grid')
    return model, elapsed, notes


def provenance(work):
    packages = {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'ngboost', 'threadpoolctl')}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'packages': packages,
            'commit': os.environ.get('GITHUB_SHA', 'local'),
            'script_sha256': reproduce.sha(Path(__file__)),
            'core_sha256': {n: reproduce.sha(work/n) for n in reproduce.CORE},
            'feature_sha256': reproduce.sha(work/'data/mfcc_delta.npz')}


def select_model(name, prepared, number):
    """Manual 3-fold search. Prefix reuse is exact for fixed seed and no early stopping."""
    from evaluation import measure
    grid = PLAN[name]['grid']
    scores = np.zeros((3, 3))
    times, notes = [], []
    for j, (x, y, valid, yy) in enumerate(prepared):
        if name == 'ngboost':
            model, seconds, w = fit_model(name, max(grid), x, y, PLAN['seed']+100*number+j)
            times.append(seconds); notes.extend(w)
            for k, value in enumerate(grid):
                scores[k, j] = np.mean([measure(yy, predict(model, xx, name, value), np.arange(10))['macro_f1'] for xx in valid])
        else:
            for k, value in enumerate(grid):
                model, seconds, w = fit_model(name, value, x, y, PLAN['seed']+100*number+j)
                times.append(seconds); notes.extend(w)
                scores[k, j] = np.mean([measure(yy, predict(model, xx, name), np.arange(10))['macro_f1'] for xx in valid])
        print(f'{name} fold={number} inner={j+1}/3 complete', flush=True)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite inner selection scores')
    return grid[int(np.argmax(scores.mean(1)))], scores, times, notes


def run_fold(work, number):
    from evaluation import grouped_folds, training_rows, scaling, transform, measure
    reproduce.check_data(work)
    with np.load(work/'data/mfcc_delta.npz') as d:
        y, groups = d['y'], d['groups']
        variants = {c: d[c] for c in CONDITIONS}
    train, test = list(grouped_folds(groups, np.arange(len(y)), 10, PLAN['seed']))[number-1]
    inner = list(grouped_folds(groups, train, 3, PLAN['seed']+number))
    prepared = []
    for tr, va in inner:
        x, yy = training_rows(variants, y, tr, PLAN['mode']); stats = scaling(x)
        prepared.append((transform(x, stats), yy, [transform(variants[c][va], stats) for c in ('clean','white_10')], y[va]))
    x, yy = training_rows(variants, y, train, PLAN['mode']); stats = scaling(x); x = transform(x, stats)
    folder = work/'comparison'/f'fold_{number:02}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        raise FileExistsError('Completed fold exists. Use a fresh work directory; never overwrite evidence.')
    record = {'plan': PLAN, 'provenance': provenance(work), 'fold': number,
              'train_people': np.unique(groups[train]).tolist(), 'test_people': np.unique(groups[test]).tolist(),
              'inner_valid_people': [np.unique(groups[v]).tolist() for _,v in inner],
              'train_n': len(train), 'test_n': len(test), 'fit_rows': len(x), 'models': {}, 'status': 'RUNNING'}
    # Balance order to reduce first-model/warm-cache bias; never run the pair concurrently.
    order = ['ngboost','lightgbm'] if number % 2 else ['lightgbm','ngboost']
    record['execution_order'] = order
    try:
        for name in order:
            start = time.perf_counter()
            chosen, scores, inner_seconds, notes = select_model(name, prepared, number)
            model, fit_seconds, w = fit_model(name, chosen, x, yy, PLAN['seed']+number); notes.extend(w)
            prediction_start = time.perf_counter()
            probabilities = {c: predict(model, transform(variants[c][test], stats), name) for c in CONDITIONS}
            predict_seconds = time.perf_counter()-prediction_start
            metrics = {c: measure(y[test], p, np.arange(10)) for c,p in probabilities.items()}
            file = folder/f'{name}.pkl'
            with file.open('wb') as f:
                pickle.dump({'estimator':model,'scaler':stats,'classes':np.arange(10),'selected':chosen,'train_people':record['train_people']}, f, protocol=5)
            with file.open('rb') as f:
                loaded = pickle.load(f)  # Only the model just created by this process.
            assert all(np.array_equal(a,b) for a,b in zip(stats,loaded['scaler']))
            delta = 0.
            for c in CONDITIONS:
                again = predict(loaded['estimator'], transform(variants[c][test], loaded['scaler']), name)
                delta = max(delta, float(np.max(np.abs(again-probabilities[c]))))
                np.testing.assert_allclose(again, probabilities[c], atol=1e-6, rtol=1e-5)
            predictions = folder/f'{name}_predictions.npz'
            np.savez_compressed(predictions, indices=test, y=y[test], groups=groups[test], classes=np.arange(10), **probabilities)
            row = {'selected': chosen, 'grid': PLAN[name]['grid'], 'inner_scores': scores.tolist(),
                   'inner_means': scores.mean(1).tolist(), 'metrics': metrics, 'warnings': sorted(set(notes)),
                   'inner_fit_seconds': inner_seconds, 'final_fit_seconds': fit_seconds,
                   'evaluation_seconds': predict_seconds, 'workflow_seconds':time.perf_counter()-start,
                   'actual_fit_calls': 4 if name=='ngboost' else 10,
                   'model_bytes':file.stat().st_size,'model_sha256':reproduce.sha(file),
                   'predictions_sha256':reproduce.sha(predictions), 'reload_max_probability_difference':delta,
                   'model_reloaded_verified':True}
            if name == 'ngboost':
                row['training_log_scores'] = model.evals_result['train']['LOGSCORE']
                row['actual_stages'] = len(model.base_models)
            record['models'][name] = row
            reproduce.save(folder/'progress.json', record)
            print(f'{name} fold={number} selected={chosen} final_fit_s={fit_seconds:.2f}', flush=True)
        record['status'] = 'PASS'
        reproduce.save(folder/'result.json', record)
    except Exception:
        record['status'] = 'FAIL'; record['error'] = traceback.format_exc()
        reproduce.save(folder/'failure.json', record)
        raise


def summarize(root):
    from evaluation import grouped_folds, measure
    paths = sorted((root/'comparison').glob('fold_*/result.json'))
    if len(paths) != 10:
        raise ValueError(f'Require all ten successful paired folds, found {len(paths)}; failed or missing folds must not be dropped.')
    docs = [json.loads(p.read_text()) for p in paths]
    assert sorted(d['fold'] for d in docs) == list(range(1,11))
    all_ids = []; references = []
    for p,d in zip(paths,docs):
        assert d['status']=='PASS' and d['plan']==PLAN
        assert set(d['models'])=={'ngboost','lightgbm'}
        assert not set(d['train_people']) & set(d['test_people'])
        assert len(d['train_people'])==54 and len(d['test_people'])==6
        assert d['train_n']==10800 and d['test_n']==1200 and d['fit_rows']==21600
        references.append((d['provenance']['script_sha256'], d['provenance']['feature_sha256'], d['provenance']['packages'], d['provenance']['core_sha256']))
        pair = []
        for name,row in d['models'].items():
            assert row['model_reloaded_verified'] and row['reload_max_probability_difference']<=1e-6
            scores=np.array(row['inner_scores']); assert scores.shape==(3,3) and np.isfinite(scores).all()
            assert row['selected']==PLAN[name]['grid'][int(np.argmax(scores.mean(1)))]
            assert np.allclose(scores.mean(1),row['inner_means'])
            pred=p.parent/f'{name}_predictions.npz'; assert reproduce.sha(pred)==row['predictions_sha256']
            with np.load(pred) as z:
                assert np.array_equal(z['classes'],np.arange(10))
                pair.append((z['indices'].copy(),z['y'].copy(),z['groups'].copy()))
                assert np.unique(z['groups']).tolist()==d['test_people']
                for c in CONDITIONS:
                    metrics=measure(z['y'],valid_probabilities(z[c],len(z['y'])),z['classes'])
                    for m in METRICS:
                        assert abs(metrics[m]-row['metrics'][c][m])<1e-12
        assert all(np.array_equal(a,b) for a,b in zip(*pair))
        all_ids.extend(pair[0][0].tolist())
    assert all(x==references[0] for x in references)
    assert sorted(all_ids)==list(range(12000)), 'Outer test recordings must appear exactly once'
    # Reconstruct group vector from predictions and independently verify exact seeded folds.
    groups=np.empty(12000,dtype=int)
    for p in paths:
        with np.load(p.parent/'ngboost_predictions.npz') as z: groups[z['indices']]=z['groups']
    splits=list(grouped_folds(groups,np.arange(12000),10,PLAN['seed']))
    for d,(tr,te) in zip(docs,splits):
        assert np.unique(groups[te]).tolist()==d['test_people']
        assert [np.unique(groups[v]).tolist() for _,v in grouped_folds(groups,tr,3,PLAN['seed']+d['fold'])]==d['inner_valid_people']
    rows=[]; paired=[]
    for c in CONDITIONS:
        for name in ('ngboost','lightgbm'):
            row={'model':name,'condition':c}
            for m in METRICS:
                a=[d['models'][name]['metrics'][c][m] for d in docs]
                row[m+'_mean']=st.mean(a);row[m+'_fold_sd']=st.stdev(a)
            rows.append(row)
        for m in METRICS:
            a=[d['models']['ngboost']['metrics'][c][m]-d['models']['lightgbm']['metrics'][c][m] for d in docs]
            paired.append({'condition':c,'metric':m,'difference':'NGBoost minus LightGBM',
                           'mean':st.mean(a),'fold_sd':st.stdev(a), 'min':min(a),'max':max(a),
                           'ngboost_wins':sum(v<0 if m=='log_loss' else v>0 for v in a),
                           'ties':sum(v==0 for v in a),'fold_differences':a})
    report=root/'reports'; report.mkdir(exist_ok=True)
    for fn,rr in [('summary.csv',rows),('paired_differences.csv',[{k:v for k,v in r.items() if k!='fold_differences'} for r in paired])]:
        with (report/fn).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    costs={};audit={}
    for name in ('ngboost','lightgbm'):
        costs[name]={k+'_mean':st.mean(d['models'][name][k] for d in docs) for k in ('final_fit_seconds','workflow_seconds','evaluation_seconds','model_bytes')}
        counts=[sum(d['models'][name]['selected']==v for d in docs) for v in PLAN[name]['grid']]
        audit[name]={'grid':PLAN[name]['grid'],'counts':counts,'middle_unique_mode':counts[1]>max(counts[0],counts[2]),
                     'warnings':sum(len(d['models'][name]['warnings']) for d in docs)}
    reproduce.save(report/'comparison.json',{'status':'PASS','scope':'MFCC+delta, white-10 augmented, 10 x 3 manual speaker CV',
                   'plan':PLAN,'folds':10,'paired':paired,'costs':costs,'parameter_audit':audit,
                   'result_sha256':{str(p.relative_to(root)):reproduce.sha(p) for p in paths},
                   'limits':['One partition; fold SD is not an independent CI.', 'Grids were fixed before new outer results; prior corpus exploration remains.',
                             'NGBoost grid selection may hit a boundary; this does not invalidate measurement but limits model tuning/course compliance.',
                             'Unequal model capacity/stage counts; results compare specified pipelines, not universal algorithm superiority.',
                             'Workflow time includes selection, prediction, serialization and verification; different fit counts due to NGBoost stage reuse.',
                             'Serialized bytes are not peak RAM; evaluation includes transforms and four batch predictions, not device latency.',
                             'No new full-data deployment model is trained; twenty outer models are comparison artifacts.']})
    lines=['# NGBoost versus LightGBM: measured comparison','',
           'MFCC+delta; clean + white-10 training; ten paired unseen-speaker folds, three inner folds. Values are means ± sample fold SD.','',
           '| Condition | NGBoost F1 | LightGBM F1 | NGBoost − LightGBM F1 | NGBoost wins |','|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        n=next(r for r in rows if r['model']=='ngboost' and r['condition']==c)
        l=next(r for r in rows if r['model']=='lightgbm' and r['condition']==c)
        p=next(r for r in paired if r['condition']==c and r['metric']=='macro_f1')
        lines.append(f"| {c} | {n['macro_f1_mean']:.4f} ± {n['macro_f1_fold_sd']:.4f} | {l['macro_f1_mean']:.4f} ± {l['macro_f1_fold_sd']:.4f} | {p['mean']:+.4f} ± {p['fold_sd']:.4f} | {p['ngboost_wins']}/10 |")
    lines+=['','## Costs and grid audit','', '| Model | Final fit s | Workflow s | Serialized model MB | Grid selection counts |','|---|---:|---:|---:|---|']
    for name in costs:
        c=costs[name]
        lines.append(f"| {name} | {c['final_fit_seconds_mean']:.2f} | {c['workflow_seconds_mean']:.2f} | {c['model_bytes_mean']/1e6:.2f} | {audit[name]['counts']} |")
    lines+=['','See summary.csv for accuracy, precision, recall and log loss; paired_differences.csv and comparison.json for all paired comparisons.',
            '', 'Results apply only to the declared pipelines. A boundary-selected grid must be disclosed; no automatic retuning uses outer test scores. Fold SD and win counts are descriptive, not independent significance tests. Different stage counts, implementation and selection reuse affect costs. No peak RAM, calibration or deployment latency claim is made.']
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PAIRED_COMPARISON_VERIFIED',report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','fold','summarize'])
    p.add_argument('--work-dir',type=Path,default=Path('.ngboost-comparison'))
    p.add_argument('--fold',type=int,choices=range(1,11))
    p.add_argument('--protocol',choices=['legacy200','extended800'],default='legacy200',
                   help='Legacy 50/100/200 grid or expanded 200/400/800 grid; keep a separate work directory.')
    a=p.parse_args()
    configure_protocol(a.protocol)
    if a.action=='plan': print(json.dumps(PLAN,indent=2));return
    work=reproduce.stage(a.work_dir)
    if a.action=='fold':
        if a.fold is None:p.error('--fold is required')
        run_fold(work,a.fold)
    else:summarize(work)


if __name__=='__main__': main()
