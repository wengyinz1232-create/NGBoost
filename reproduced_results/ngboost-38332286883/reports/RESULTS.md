# NGBoost versus LightGBM: measured comparison

MFCC+delta; clean + white-10 training; ten paired unseen-speaker folds, three inner folds. Values are means ± sample fold SD.

| Condition | NGBoost F1 | LightGBM F1 | NGBoost − LightGBM F1 | NGBoost wins |
|---|---:|---:|---:|---:|
| clean | 0.9317 ± 0.0250 | 0.9640 ± 0.0158 | -0.0323 ± 0.0112 | 0/10 |
| white_20 | 0.8873 ± 0.0391 | 0.9357 ± 0.0238 | -0.0484 ± 0.0177 | 0/10 |
| white_10 | 0.8778 ± 0.0474 | 0.9373 ± 0.0268 | -0.0595 ± 0.0220 | 0/10 |
| pink_10 | 0.8569 ± 0.0398 | 0.9071 ± 0.0399 | -0.0502 ± 0.0145 | 0/10 |

## Costs and grid audit

| Model | Final fit s | Workflow s | Serialized model MB | Grid selection counts |
|---|---:|---:|---:|---|
| ngboost | 3074.39 | 9058.52 | 10.34 | [0, 0, 10] |
| lightgbm | 8.24 | 67.67 | 0.66 | [3, 6, 1] |

See summary.csv for accuracy, precision, recall and log loss; paired_differences.csv and comparison.json for all paired comparisons.

Results apply only to the declared pipelines. A boundary-selected grid must be disclosed; no automatic retuning uses outer test scores. Fold SD and win counts are descriptive, not independent significance tests. Different stage counts, implementation and selection reuse affect costs. No peak RAM, calibration or deployment latency claim is made.
