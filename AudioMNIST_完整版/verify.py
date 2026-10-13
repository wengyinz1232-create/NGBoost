"""Executable checks, independent arithmetic, and recorded-artifact validation."""
import argparse
import json
import pickle
import numpy as np
from common import ROOT, SEED, REPRESENTATIONS, MODELS, MODES, CONDITIONS, sha, save_json
from features import noisy_audio, audio_features, feature_names
from evaluation import grouped_folds, measure, scaling, transform, proba, training_rows


def unit_checks():
    p = np.array([[.8,.2],[.3,.7],[.1,.9],[.2,.8]])
    score = measure(np.array([0,0,1,1]),p,np.array([0,1]))
    assert score['confusion'] == [[1,1],[0,2]]
    assert abs(score['accuracy']-.75)<1e-12 and abs(score['macro_f1']-11/15)<1e-12
    assert np.allclose(transform(np.array([[101.,102.]]),scaling(np.array([[1.,2.],[3.,4.]]))),[[99.,99.]])
    waves = np.tile(np.sin(np.arange(16000)/17),(3,1)).astype(np.float32)
    for pink in (False,True):
        for snr in (10,20):
            noisy = noisy_audio(waves,snr,123,pink)
            measured = 10*np.log10(np.mean(waves**2,1)/np.mean((noisy-waves)**2,1))
            assert np.allclose(measured,snr,atol=1e-4)
    features = audio_features(waves)
    for rep,n in [('logmel',256),('mfcc_delta',320)]:
        assert features[rep].shape == (3,n) and len(set(feature_names(rep))) == n
        assert np.isfinite(features[rep]).all()
    print('UNIT_CHECKS_PASS',flush=True)


def verify():
    unit_checks()
    inventory = json.loads((ROOT/'data'/'inventory.json').read_text(encoding='utf-8'))
    assert inventory['signature']['raw_sha256'] == sha(ROOT/'data_raw'/'audiomnist.zip')
    assert inventory['signature']['frontend_sha256'] == sha(ROOT/'features.py')
    assert inventory['signature']['prepare_sha256'] == sha(ROOT/'prepare_data.py')
    reference = None; caches = {}
    for rep in REPRESENTATIONS:
        assert sha(ROOT/'data'/f'{rep}.npz') == inventory['cache_sha256'][rep]
        with np.load(ROOT/'data'/f'{rep}.npz') as archive: cache = {k:archive[k] for k in archive.files}
        caches[rep] = cache
        identity = [cache[k] for k in ('y','groups','ids')]
        if reference is not None: assert all(np.array_equal(a,b) for a,b in zip(identity,reference))
        else: reference = identity
        y,g,ids = identity
        assert len(y)==len(set(ids))==12000 and len(np.unique(g))==60
        assert np.array_equal(np.bincount(y),np.repeat(1200,10))
        for c in CONDITIONS:
            assert cache[c].shape==(12000,len(feature_names(rep))) and np.isfinite(cache[c]).all()
    y,groups,ids = reference
    expected_folds = list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    assert sorted(np.concatenate([test for _,test in expected_folds]).tolist()) == list(range(len(y)))
    paths = sorted((ROOT/'runs').glob('*/*/result.json'))
    main_paths = list((ROOT/'runs'/'main').glob('*/result.json'))
    expected = {f'{r}__{m}__{s}' for r in REPRESENTATIONS for m in MODELS for s in MODES}
    assert {p.parent.name for p in main_paths} == expected
    verified = 0; grid_audit = []
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8')); config=result['config']; cache=caches[config['representation']]
        assert len(result['folds'])==10 and sorted(r['fold'] for r in result['folds'])==list(range(1,11))
        assert 'full_fit' in result and result['full_fit']['test_n']==0
        for name,digest in config['signature'].items():
            source = ROOT/'data'/name if name.endswith('.npz') else ROOT/name
            assert sha(source)==digest
        for row in result['folds']+[result['full_fit']]:
            number = row['fold']
            train,test = expected_folds[number-1] if number else (np.arange(len(y)),np.array([],dtype=int))
            assert row['train_people']==np.unique(groups[train]).tolist()
            assert row['test_people']==np.unique(groups[test]).tolist()
            assert row['train_n']==len(train) and row['test_n']==len(test)
            assert row['fit_rows']==len(train)*(2 if config['mode']=='augmented' else 1)
            assert not set(row['train_people']) & set(row['test_people'])
            inner = list(grouped_folds(groups,train,3,SEED+number))
            assert row['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in inner]
            assert sorted(sum(row['inner_valid_people'],[]))==row['train_people']
            scores = np.asarray(row['inner_scores']); assert scores.shape==(3,3)
            assert np.allclose(scores.mean(1),row['inner_means'])
            assert row['selected']==config['grid'][int(np.argmax(scores.mean(1)))]
            model_path=path.parent/'models'/f'fold_{number:02}.pkl'
            assert row['model_sha256']==sha(model_path)
            with open(model_path,'rb') as stream: bundle=pickle.load(stream)
            assert bundle['train_people']==row['train_people'] and bundle['selected']==row['selected']
            xfit,_=training_rows(cache,y,train,config['mode'])
            expected_scaler=scaling(xfit)
            assert all(np.array_equal(a,b) for a,b in zip(bundle['scaler'],expected_scaler))
            del xfit
            if not number:
                # Full-data model can predict, but is not evaluated on its training set.
                p=proba(bundle['estimator'],transform(cache['clean'][:3],bundle['scaler']),bundle['classes'])
                assert p.shape==(3,10) and np.allclose(p.sum(1),1)
                continue
            with np.load(path.parent/f'predictions_{number:02}.npz') as predicted:
                assert np.array_equal(predicted['indices'],test) and np.array_equal(predicted['y'],y[test])
                for c in CONDITIONS:
                    p=predicted[c]
                    assert p.shape==(1200,10) and np.allclose(p.sum(1),1)
                    assert np.isfinite(p).all() and np.all(p>=0) and np.all(p<=1)
                    actual=measure(y[test],p,predicted['classes'])
                    assert actual==row['metrics'][c]
                fresh=proba(bundle['estimator'],transform(cache['clean'][test[:8]],bundle['scaler']),bundle['classes'])
                assert np.allclose(fresh,predicted['clean'][:8],atol=1e-6,rtol=1e-5)
            verified+=1
        counts=[sum(r['selected']==v for r in result['folds']) for v in config['grid']]
        grid_audit.append({'configuration':str(path.relative_to(ROOT)), 'counts':counts,
            'middle_unique_mode':counts[1]>max(counts[0],counts[2]), 'middle_majority':counts[1]>5})
    save_json(ROOT/'VERIFICATION.json',{'status':'PASS','outer_models_verified':verified,
        'full_models_verified':len(paths),'checks':['known metric arithmetic','training-only scaling',
        'white and pink SNR','raw and code hashes','cache dimensions and identities',
        '10x3 speaker isolation','tuning argmax','augmentation row counts',
        'every saved probability matrix reproduces metrics','model reload predictions'],
        'grid_requirement_audit':grid_audit,
        'note':'PASS means integrity checks pass, not automatic compliance with the grid midpoint or topic approval.'})
    print('VERIFY_PASS',verified,'outer models;',len(paths),'full-data models',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit-only',action='store_true')
    if p.parse_args().unit_only:unit_checks()
    else:verify()
