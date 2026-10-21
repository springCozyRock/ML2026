# Node 2 executor report

## Idea

Cross-fitted categorical target encoding and training category frequencies with a regularized LightGBM regressor, intended to complement the cached Lasso/SVR/residual-XGB blend.

## Changes and implementation choices

Only `solution.py` changed. Existing extended feature engineering is fitted on each outer training fold. `TargetEncoder(target_type="continuous", smooth=20, cv=5, shuffle=True)` uses `fit_transform` for internal cross-fitted training values and outer-training-only `transform` for validation/test. Category frequencies, imputations, and frontage medians use outer training features only. No validation labels or external data are read by the candidate.

The one fixed LightGBM configuration uses 1600 trees, learning rate 0.025, 15 leaves, minimum child samples 20, L1 0.1, L2 8, row/column fractions 0.85, and eight CPU threads. No early stopping or parameter sweep was performed. The protected evaluator supplies the rare-row gate and fixed 0.2 blend.

## Validation

The cheap protected baseline reproduced 0.10432475735803212. Small representative checks passed finite output shapes, an unseen Neighborhood category, deterministic repeated predictions, and protected hashes. `git diff --check` passed. One full five-fold B_dev evaluation completed in 11.253635168075562 seconds.

The evaluation verified 1460 OOF rows, 1459 submission rows, valid submission columns and IDs, positive finite prices, and protected hashes before/after. No B_test evaluation or Kaggle submission was performed.

## Baseline vs result

- Baseline: 0.10432475735803212
- Absolute primary score: **0.10533124408155023**
- Increase in RMSE: 0.00100648672351811 (about 0.965%)
- Fold RMSE: 0.10409424561748096, 0.11078758736443844, 0.08951704499110448, 0.11867176833723252, 0.10132898632624728

## Analysis and insight

The candidate was correctly active but worsened the fixed blend in four of five folds. Training-only cross-fitted target encoding with this regularized boosting configuration did not provide a useful complementary correction to the existing blend. Keep the baseline over this node; the result does not establish that all target-encoded models would fail.

## Artifacts and code

`results/node2/` contains `oof.csv`, `submission.csv`, `metrics.json`, `source.diff`, and this report. Source commit: `d1f0433` on `codex/arbor-node-2`. The submission is an experiment artifact only.
