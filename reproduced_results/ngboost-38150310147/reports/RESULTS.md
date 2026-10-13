# NGBoost versus LightGBM: measured comparison

MFCC+delta; clean + white-10 training; ten paired unseen-speaker folds, three inner folds. Values are means ± sample fold SD.

| Condition | NGBoost F1 | LightGBM F1 | NGBoost − LightGBM F1 | NGBoost wins |
|---|---:|---:|---:|---:|
| clean | 0.9275 ± 0.0282 | 0.9640 ± 0.0158 | -0.0365 ± 0.0154 | 0/10 |
| white_20 | 0.8772 ± 0.0380 | 0.9357 ± 0.0238 | -0.0586 ± 0.0172 | 0/10 |
| white_10 | 0.8689 ± 0.0478 | 0.9373 ± 0.0268 | -0.0684 ± 0.0233 | 0/10 |
| pink_10 | 0.8466 ± 0.0374 | 0.9071 ± 0.0399 | -0.0605 ± 0.0168 | 0/10 |

## Costs and grid audit

| Model | Final fit s | Workflow s | Serialized model MB | Grid selection counts |
|---|---:|---:|---:|---|
| ngboost | 740.44 | 2181.04 | 2.60 | [0, 0, 10] |
| lightgbm | 8.20 | 67.30 | 0.66 | [3, 6, 1] |

See summary.csv for accuracy, precision, recall and log loss; paired_differences.csv and comparison.json for all paired comparisons.

Results apply only to the declared pipelines. A boundary-selected grid must be disclosed; no automatic retuning uses outer test scores. Fold SD and win counts are descriptive, not independent significance tests. Different stage counts, implementation and selection reuse affect costs. No peak RAM, calibration or deployment latency claim is made.
