# NGBoost paired training experiment

This is an actual training entry point, separate from the original twelve configurations. Its scope is **MFCC+delta with clean + white-10 augmentation**, compared against the current seven-leaf LightGBM–GOSS configuration. Results do not cover clean-only training or Log-Mel NGBoost.

## Completed extended800 result

[Expanded results and retained predictions](../reproduced_results/ngboost-38332286883/NGBOOST_RESULTS_ZH.md) use training commit `1c5ed1eb08c90922f3be7def6672ede6cced73bb`. All ten paired-fold jobs in [run 38332286883](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38332286883) succeeded. The final summarize job never started because of GitHub account payment/spending limits; the overall Actions conclusion remains failure. The ten per-fold artifacts were downloaded and the unchanged summarizer completed locally. There is no aggregate Actions artifact for this run. No training was rerun and no billing setting was changed.

All ten inner searches selected 800. Relative to legacy200, mean F1 gains were 0.00419 / 0.01013 / 0.00886 / 0.01025 for clean / W20 / W10 / P10. LightGBM still won all ten folds in every condition. NGBoost mean final fitting cost was 3074.39 seconds per fold versus 8.24 seconds for paired LightGBM. Clean and white-noise log loss slightly worsened; pink-noise log loss slightly improved. Same feature hashes and speaker splits were verified across runs; between-run timing is not hardware-controlled. The boundary-selection and repeated-exploration limitations remain.

All saved prediction metrics were recomputed. All 20 model hashes were checked and models reloaded locally on 16 held-out recordings per model and condition, in addition to the cloud reload checks. Models are retained in the local deliverable ZIP, while the Git directory retains result records and predictions. Rebuild metrics without audio or model files:

```bash
python compare_ngboost.py summarize --work-dir reproduced_results/ngboost-38332286883 --protocol extended800
```

## Historical completed run

[Run 38150310147](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38150310147) completed all ten paired folds at training commit `e2c58ebc1a800900d10051eda1f0f24e5d40397e`. [Measured report and retained evidence](../reproduced_results/ngboost-38150310147/NGBOOST_RESULTS_ZH.md) include predictions so metrics remain verifiable after Actions artifacts expire. Model files are in the downloadable workflow artifact, not committed to Git.

Pink-noise F1 was 0.8466 ± 0.0374 for NGBoost versus 0.9071 ± 0.0399 for LightGBM, with LightGBM winning all ten paired folds. Final-fit means were 740.44 versus 8.20 seconds. NGBoost selected the 200-stage upper boundary in every fold; its grid fails the course midpoint criterion and cannot establish fully tuned algorithmic inferiority. Neither model produced recorded training warnings, and NGBoost's pink-noise fold SD was slightly smaller. Thus the run supports a performance/cost disadvantage for the declared NGBoost pipeline, not a claim of numerical instability or universal inferiority.

## Expanded stage search: 200 / 400 / 800

The `extended800` protocol keeps learning rate 0.03, the existing base trees, all sampling settings, speaker folds and the clean/W10 mean macro-F1 selection objective unchanged. Each of the three inner folds fits one 800-stage trajectory and evaluates prefixes 200, 400 and 800. The best mean inner score selects the final outer refit's stage count; ties select the smaller candidate. The ten final NGBoost models are therefore **not forced to use 800 stages**. The same LightGBM L2 search is repeated on each paired runner.

This is a follow-up after the original upper-bound result and a local one-split pilot. It is not independent confirmation on previously unexamined data. Do not adjust the new grid using its outer scores. Keep the legacy results, report selected counts and all paired results, and distinguish final-fit time from complete tuning time.

Use a new runtime directory and append `--protocol extended800` to every `plan`, `fold` and `summarize` command. For example:

```bash
python reproduce.py prepare --work-dir .ngboost-extended800
python compare_ngboost.py plan --protocol extended800
python compare_ngboost.py fold --work-dir .ngboost-extended800 --fold 1 --protocol extended800
# Repeat folds 2 through 10, then:
python compare_ngboost.py summarize --work-dir .ngboost-extended800 --protocol extended800
```

The workflow dispatch defaults to `extended800`. Select `legacy200` to reproduce the original 50/100/200 experiment; the CLI defaults to that historical protocol so existing saved-result commands remain valid. New runs have separate Actions artifacts and result directories. Per-fold timeout is 240 minutes to accommodate the larger budget; the same ten-fold concurrency is retained. No course-report source is changed by this experiment.

## Historical fixed protocol (`legacy200`)

- Same 12,000 recordings, seed 20261009, ten speaker-disjoint outer folds and three inner folds as the original experiment. Each outer fold has 54 training and six unseen test speakers.
- Both models select their parameter using the mean of clean and white-10 inner macro-F1. Pink noise and outer test labels never select a stage or parameter.
- NGBoost 0.5.11: ten-class categorical distribution, natural gradient, depth-3 regression trees, minimum leaf size 20, learning rate 0.03, row and column sampling 0.5. Tune stages over **50, 100, 200**.
- LightGBM 4.7.0: 70 rounds, seven leaves, learning rate 0.08, GOSS rates 0.2/0.1, minimum leaf size 20. Tune L2 over **0, 0.1, 1**, exactly as the existing MFCC configuration.
- The stage grid is a finite computational budget, not equal model capacity or proof of optimal NGBoost tuning. Do not adjust it using outer results. Report boundary selections and midpoint-audit failures honestly.
- For each NGBoost inner fold, train 200 stages once and evaluate the three prefixes. With fixed seeds and no early stopping this is equivalent to independently fitting each stage count; a test verifies this equivalence. Three trajectories plus one final refit use four actual fit calls per outer fold. LightGBM uses nine inner fits plus one refit. Both evaluate nine inner candidate scores.
- Odd folds run NGBoost first; even folds run LightGBM first. A paired fold runs sequentially on the same runner. Both have BLAS limited to one thread; LightGBM can use two tree-building threads, while this pinned NGBoost implementation fits base learners serially. Workflow times describe these implementations, not an intrinsic algorithmic speed ratio.
- Save and reload all twenty outer models, validate probabilities, save outer predictions and recompute metrics. No additional full-data deployment models are fitted. Keep the original twelve-configuration results intact.

## Run locally

Use a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements-ngboost.txt
python -m unittest discover -s tests -v
python reproduce.py prepare --work-dir .ngboost-comparison
python compare_ngboost.py plan
python compare_ngboost.py fold --work-dir .ngboost-comparison --fold 1
# Repeat --fold 2 through --fold 10, then:
python compare_ngboost.py summarize --work-dir .ngboost-comparison
```

The fold command refuses to overwrite a completed fold. Failed folds leave `failure.json`; no summary is produced by silently dropping them. Use a fresh directory after changing source or protocol.

## Run on GitHub

Actions → **NGBoost paired comparison** → Run workflow. The workflow prepares features once, runs ten independent paired-fold jobs (at most ten concurrently), then validates and summarizes all folds. No paid service is enabled by the workflow. It consumes normal Actions time and stores evidence for 14 days.

The `ngboost-comparison-evidence` artifact contains `reports/summary.csv`, `paired_differences.csv`, `comparison.json`, `RESULTS.md`, plus per-fold results, predictions and models. Only load pickle files from your own trusted run. A failed job still attempts to upload partial evidence.

Final-fit time is separate from total workflow time. Workflow time includes search, scoring, refitting, outer evaluation, serialization and reload checks, but not shared feature extraction. Evaluation timing includes transforms and four batch predictions, not device-level per-request latency. Serialized model size is not peak RAM. Differences are NGBoost minus LightGBM; lower log loss is better. Fold SD and win counts are descriptive because the training sets overlap.

## Interpretation rules

Report measured performance even if it favors NGBoost or fails to show a clear difference. A small difference or mixed fold wins cannot establish universal superiority. A boundary-selected stage grid limits any claim that the NGBoost family itself is inferior. Poorer scores under this budget could reflect underfitting, settings or computation limits rather than an inherent defect. Check loss and probability behavior before claiming numerical instability. Neither log loss nor a categorical distribution alone proves calibration or out-of-distribution uncertainty quality.

Methods: [NGBoost, ICML 2020](https://proceedings.mlr.press/v119/duan20a.html), [NGBoost classification](https://stanfordmlgroup.github.io/ngboost/1-useage.html), [stage selection](https://stanfordmlgroup.github.io/ngboost/2-tuning.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
