"""Protected House Prices development evaluator; Python 3.8 compatible."""
import argparse
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / "protected_manifest.json").read_text())
    for path, expected in manifest.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Protected file changed: " + path)
    return True

def rare(frame):
    return ((frame["GrLivArea"] > 4000) & (frame["SaleCondition"] == "Partial")).to_numpy()

def prediction(value, size, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(name + " must be a finite vector of length " + str(size))
    if not np.isfinite(np.exp(arr)).all() or (np.exp(arr) <= 0).any():
        raise ValueError(name + " must imply finite positive prices")
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--node-id", default="baseline")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.node_id):
        raise ValueError("Unsafe node ID")
    started = time.time()
    verify()
    cfg = json.loads((ROOT / "config.json").read_text())
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    cached = pd.read_csv(ROOT / "baseline/oof.csv")
    submission = pd.read_csv(ROOT / "baseline/submission.csv")
    if len(train) != 1460 or len(test) != 1459:
        raise ValueError("Unexpected data size")
    if not train.Id.is_unique or not test.Id.is_unique:
        raise ValueError("Duplicate data IDs")
    if not np.array_equal(cached.Id.to_numpy(), train.Id.to_numpy()):
        raise ValueError("Baseline OOF ID mismatch")
    if list(submission.columns) != ["Id", "SalePrice"] or not np.array_equal(submission.Id, test.Id):
        raise ValueError("Baseline submission schema/ID mismatch")
    y = np.log(train.SalePrice.to_numpy(dtype=float))
    features = train.drop(columns=["SalePrice"])
    base_oof = prediction(cached.prediction_log.to_numpy(), len(train), "baseline OOF")
    base_test = prediction(np.log(submission.SalePrice.to_numpy()), len(test), "baseline test")
    oof = base_oof.copy()
    fold_test = []
    fold_values = []
    gate_records = []
    candidate_used = None
    if not args.baseline:
        spec = importlib.util.spec_from_file_location("solution", ROOT / "solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    splitter = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_state"])
    try:
        for fold, (ti, vi) in enumerate(splitter.split(features), 1):
            train_gate = rare(features.iloc[ti])
            ordinary_ti = ti[~train_gate]
            valid_gate = rare(features.iloc[vi])
            expert_indices = ti[train_gate]
            expert_log = float(y[expert_indices].mean()) if len(expert_indices) else None
            if expert_log is None and (valid_gate.any() or rare(test).any()):
                raise RuntimeError("Rare gate has no fold training support")
            result = None if args.baseline else module.fit_predict(
                features.iloc[ordinary_ti].copy().reset_index(drop=True),
                y[ordinary_ti].copy(),
                features.iloc[vi].copy().reset_index(drop=True),
                test.copy(), cfg["random_state"] + fold)
            used = result is not None
            if candidate_used is not None and candidate_used != used:
                raise ValueError("Candidate must return consistently across folds")
            candidate_used = used
            if used:
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise ValueError("Expected pair of validation and test log predictions")
                val = prediction(result[0], len(vi), "candidate validation").copy()
                test_pred = prediction(result[1], len(test), "candidate test").copy()
                val[valid_gate] = expert_log
                test_pred[rare(test)] = expert_log
                weight = cfg["blend_weight"]
                oof[vi] = (1 - weight) * base_oof[vi] + weight * val
                fold_test.append(test_pred)
            fold_values.append(float(np.sqrt(np.mean((y[vi] - oof[vi]) ** 2))))
            if valid_gate.any():
                gate_records.append({"fold": fold, "expert_train_ids": features.iloc[expert_indices].Id.tolist(),
                    "validation_ids": features.iloc[vi[valid_gate]].Id.tolist(), "prediction_log": expert_log})
        test_log = base_test.copy()
        if candidate_used:
            test_log = (1 - cfg["blend_weight"]) * base_test + cfg["blend_weight"] * np.mean(fold_test, axis=0)
        prediction(oof, len(train), "final OOF")
        prediction(test_log, len(test), "final test")
        score = float(np.sqrt(np.mean((y - oof) ** 2)))
        out = Path(args.output) if args.output else ROOT / "results" / args.node_id
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": train.Id, "prediction_log": oof, "residual_log": y-oof}).to_csv(out / "oof.csv", index=False)
        final_sub = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(test_log)})
        final_sub.to_csv(out / "submission.csv", index=False)
        verify()
        metrics = {"node_id": args.node_id, "score": score, "metric": cfg["metric"],
            "fold_rmse": fold_values, "duration_seconds": time.time() - started,
            "candidate_used": bool(candidate_used), "blend_weight": cfg["blend_weight"] if candidate_used else 0.0,
            "oof_rows": len(train), "submission_rows": len(test), "schema_valid": True,
            "positive_finite_prices": True, "protected_hashes_verified": True,
            "rare_gate_folds": gate_records, "pristine_local_heldout": False}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("score: {:.17g}".format(score))
        print(json.dumps(metrics))
    finally:
        verify()

if __name__ == "__main__":
    main()
