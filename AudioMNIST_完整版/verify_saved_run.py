"""Verify one completed run using saved feature cache and prediction files.

Usage: python verify_saved_run.py PATH_TO_RESULT_JSON --out AUDIT.json
Checks speaker splits, nested tuning selection, and all saved metrics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, SEED, CONDITIONS
from evaluation import grouped_folds, measure


def verify(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    config = doc["config"]
    with np.load(ROOT/'data'/f'{config["representation"]}.npz') as cache:
        y, groups = cache["y"], cache["groups"]
    assert len(y) == config["samples"] and config["outer_folds"] == 10 and config["inner_folds"] == 3
    expected = list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    rows = {row["fold"]: row for row in doc["folds"]}
    assert set(rows) == set(range(1,11)) and "full_fit" in doc
    checked = 0
    for fold in range(1,11):
        row = rows[fold]; train,test = expected[fold-1]
        assert row["train_people"] == np.unique(groups[train]).tolist()
        assert row["test_people"] == np.unique(groups[test]).tolist()
        assert not set(row["train_people"]) & set(row["test_people"])
        inner = list(grouped_folds(groups,train,3,SEED+fold))
        assert row["inner_valid_people"] == [np.unique(groups[v]).tolist() for _,v in inner]
        scores = np.asarray(row["inner_scores"])
        assert scores.shape == (3,3)
        assert row["selected"] == config["grid"][int(np.argmax(scores.mean(1)))]
        with np.load(path.parent/f"predictions_{fold:02}.npz") as stored:
            assert np.array_equal(stored["indices"],test)
            assert np.array_equal(stored["y"],y[test])
            for condition in CONDITIONS:
                probabilities = stored[condition]
                assert probabilities.shape == (len(test),10)
                assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1)
                assert measure(y[test],probabilities,stored["classes"]) == row["metrics"][condition]
                checked += 1
    counts = [sum(row["selected"] == value for row in rows.values()) for value in config["grid"]]
    return {"status":"PASS", "configuration":config, "outer_folds_verified":10,
            "condition_metrics_verified":checked, "selected_counts":counts,
            "middle_unique_mode":counts[1] > max(counts[0],counts[2]),
            "note":"PASS checks saved evidence, not assignment compliance or independent generalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.result_json)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("VERIFY_SAVED_RUN_PASS",result["condition_metrics_verified"])


if __name__ == "__main__":
    main()
