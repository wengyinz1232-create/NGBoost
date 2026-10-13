import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import compare_ngboost as comparison


class PairedProtocolTests(unittest.TestCase):
    def tearDown(self):
        comparison.configure_protocol('legacy200')

    def test_extended_protocol_preserves_model_and_selection_settings(self):
        import copy
        original=copy.deepcopy(comparison.PLAN)
        comparison.configure_protocol('extended800')
        self.assertEqual(comparison.PLAN['ngboost']['grid'],[200,400,800])
        self.assertEqual(comparison.PLAN['selection'],original['selection'])
        self.assertEqual(comparison.PLAN['lightgbm'],original['lightgbm'])
        for k,v in original['ngboost'].items():
            if k!='grid': self.assertEqual(comparison.PLAN['ngboost'][k],v)
        comparison.configure_protocol('legacy200')
        self.assertEqual(comparison.PLAN,original)

    def test_extended_summary_recomputes_metrics_and_rejects_tampering(self):
        comparison.configure_protocol('extended800')
        self.test_full_summary_recomputes_metrics_and_rejects_tampering()

    def test_full_summary_recomputes_metrics_and_rejects_tampering(self):
        import json
        import reproduce
        with tempfile.TemporaryDirectory() as tmp:
            root = reproduce.stage(Path(tmp))
            from evaluation import grouped_folds, measure
            groups=np.repeat(np.arange(60),200); y=np.tile(np.arange(10),1200)
            for i,(tr,te) in enumerate(grouped_folds(groups,np.arange(12000),10,comparison.PLAN['seed']),1):
                folder=root/'comparison'/f'fold_{i:02}'; folder.mkdir(parents=True)
                inner=list(grouped_folds(groups,tr,3,comparison.PLAN['seed']+i))
                d={'fold':i,'status':'PASS','plan':comparison.PLAN,
                   'train_people':np.unique(groups[tr]).tolist(),'test_people':np.unique(groups[te]).tolist(),
                   'inner_valid_people':[np.unique(groups[v]).tolist() for _,v in inner],
                   'train_n':10800,'test_n':1200,'fit_rows':21600,
                   'provenance':{'script_sha256':'test','feature_sha256':'test','packages':{},'core_sha256':{}},'models':{}}
                p=np.full((1200,10),.1)
                for name in ('ngboost','lightgbm'):
                    file=folder/f'{name}_predictions.npz'
                    np.savez_compressed(file,indices=te,y=y[te],groups=groups[te],classes=np.arange(10),**{c:p for c in comparison.CONDITIONS})
                    d['models'][name]={'selected':comparison.PLAN[name]['grid'][0],
                        'inner_scores':[[.3]*3,[.2]*3,[.1]*3],'inner_means':[.3,.2,.1],
                        'model_reloaded_verified':True,'reload_max_probability_difference':0,
                        'predictions_sha256':reproduce.sha(file),'warnings':[],
                        'metrics':{c:measure(y[te],p,np.arange(10)) for c in comparison.CONDITIONS},
                        'final_fit_seconds':1,'workflow_seconds':2,'evaluation_seconds':.1,'model_bytes':100}
                reproduce.save(folder/'result.json',d)
            comparison.summarize(root)
            summary=json.loads((root/'reports/comparison.json').read_text())
            self.assertTrue(all(p['mean']==0 and p['ties']==10 for p in summary['paired']))
            file=root/'comparison/fold_01/result.json'
            d=json.loads(file.read_text());d['models']['ngboost']['metrics']['pink_10']['macro_f1']+=.1
            reproduce.save(file,d)
            with self.assertRaises(AssertionError):comparison.summarize(root)

    def test_rejects_missing_folds_instead_of_reporting_survivors(self):
        # Import evaluation from the isolated runtime as the CLI does.
        import reproduce
        with tempfile.TemporaryDirectory() as tmp:
            root = reproduce.stage(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'all ten'):
                comparison.summarize(root)

    def test_rejects_invalid_probability_evidence(self):
        good = np.full((2,10), .1)
        comparison.valid_probabilities(good, 2)
        for bad in [good[:, :9], good*2, np.full((2,10), np.nan)]:
            with self.assertRaises(ValueError):
                comparison.valid_probabilities(bad, 2)

    @unittest.skipUnless(importlib.util.find_spec('ngboost'), 'optional experiment dependency')
    def test_prefix_predictions_equal_independent_short_fit_and_reload(self):
        import pickle
        rng=np.random.default_rng(123)
        x=rng.normal(size=(400,12)); y=np.tile(np.arange(10),40)
        long=comparison.ng_model(4,7); long.fit(x,y)
        short=comparison.ng_model(2,7); short.fit(x,y)
        np.testing.assert_allclose(comparison.predict(long,x[:20],'ngboost',2),
                                   comparison.predict(short,x[:20],'ngboost'),atol=1e-12)
        restored=pickle.loads(pickle.dumps(long))
        np.testing.assert_allclose(comparison.predict(restored,x[:20],'ngboost'),
                                   comparison.predict(long,x[:20],'ngboost'),atol=1e-12)


if __name__=='__main__': unittest.main()
