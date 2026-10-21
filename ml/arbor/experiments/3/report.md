# Node 3: robust residual correction

## Idea

Test whether limiting the influence of large training residuals with pseudo-Huber loss improves the fixed blend with the existing House Prices baseline.

## Changes

Only `solution.py` changed. The candidate fits the protected legacy engineered/Box-Cox features on each training fold, then RobustScaler and Lasso. A pseudo-Huber XGBoost model learns the Lasso training residuals and adds its correction to validation and test log predictions.

## Implementation choices

Lasso alpha 0.0006 and maximum 15,000 iterations. XGBoost objective `reg:pseudohubererror`, huber slope 0.1, base score 0, 1,000 fixed rounds, learning rate 0.025, depth 3, minimum child weight 2, subsample 0.8, column subsample 0.7, L1 penalty 0.05, L2 penalty 2, CPU hist and 8 threads. No early stopping, sweep, package installation, validation labels, network calls, cached prediction reads, or fold-label selection inside the candidate.

The protected evaluator uses all 1,460 validation rows across the fixed five folds (seed 73), excludes the two feature-defined rare rows from ordinary candidate training, applies its rare expert gate, and blends candidate log predictions with the baseline at weight 0.2.

## Baseline vs result

Baseline RMSE: **0.10432475735803212**. Candidate blend RMSE: **0.10419447655090831**. Absolute improvement: **0.00013028080712381** (about 0.125%). Three of five folds improved; folds 1 and 5 regressed slightly. Evaluation duration: **7.8927 seconds**.

## Validation

Baseline verification reproduced the cached score exactly. Small representative checks passed for finite prediction shapes, positive implied prices, unseen categories, missing frontage, unchanged caller input, label-column rejection, and protected file hashes. One full five-fold evaluation completed; it confirmed 1,460 OOF rows, 1,459 submission rows, valid schema, finite positive prices, and matching protected hashes before and after evaluation.

## Analysis and insights

The fixed robust residual blend gives a small development improvement. This is evidence for the complete candidate; there is no matched squared-error ablation to isolate the loss function's contribution. Development labels were previously used in model work, so this score is not a pristine heldout estimate. No Kaggle evaluation or submission was performed by this executor.

## Artifacts

`metrics.json`, `oof.csv`, `submission.csv`, `training.log`, `source.diff`, and this report are stored alongside each other in `results/node3/`. Candidate source is committed on `codex/arbor-node-3` at `03fedda6d71288b2ace37ad3670d2306a3ef77b1`. Server mirror: `/mnt/data/bts/repos/homework/ml/arbor_runs/20261013/node3`.

```json
{
  "score": 0.10419447655090831,
  "insight": "The fixed Lasso plus pseudo-Huber residual blend improved development RMSE by 0.00013028 (about 0.125%), with improvement in three of five folds. This does not isolate robust loss versus the other candidate choices, and no pristine heldout or Kaggle result was measured.",
  "result": "One fixed 5-fold development evaluation passed protected hashes and prediction schema checks; blended RMSE was 0.10419447655090831 versus baseline 0.10432475735803212.",
  "code_ref": "codex/arbor-node-3"
}
```
