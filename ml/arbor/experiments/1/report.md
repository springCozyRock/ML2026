# Node 1: spline Ridge

## Idea
Smooth quality, floor-area, and age effects may complement the baseline ensemble.

## Changes
Only solution.py changed. Candidate uses the protected legacy extended/log features, plus cubic quantile splines (five knots) on 14 quality/area/age/bath numeric features. Training-fold median imputation and StandardScaler feed one Ridge(alpha=20, cholesky) model. No parameter sweep, validation labels, Kaggle, package installs or evaluator modifications.

## Implementation Choices
All preprocessing is fitted on ordinary fold training; the evaluator removes two feature-defined anomalies only from candidate training, preserves all validation rows and applies protected leave-one-another-out rare-sale treatment. Eight CPU threads maximum. Fixed candidate blending weight is 0.2.

## Baseline vs Result
Baseline all-row OOF RMSE: 0.10432475735803212.
Candidate fixed blend all-row OOF RMSE: 0.10576916790256756.
Difference: +0.00144441054453544 (worse). Full evaluation: 2.848785638809204 seconds on server.

## Validation
Server cached baseline reproduced exactly. Small representative prediction check passed finite values, vector shapes, retained IDs, and protected hashes. Full KFold(5,shuffle=True,random_state=73) scored all 1460 OOF rows; 1459 test predictions are finite and positive, submission IDs/schema valid. Hashes verified before/after evaluation and again locally. Fold RMSE: 0.1026059494153676, 0.11060079347497945, 0.08930885778062395, 0.11985512991542543, 0.1040852716581956.

## Score
0.10576916790256756

## Analysis
One fold improved slightly, while four worsened. The additive smooth candidate at the prescribed fixed weight adds error overall; retain the protected baseline. There is no pristine local heldout, and no online submission was made.

## Insights
A smooth spline Ridge candidate did not add sufficient complementary signal to the existing ensemble at fixed0.2 blending. This is a valid negative result on historical development CV, not independent generalization evidence.
