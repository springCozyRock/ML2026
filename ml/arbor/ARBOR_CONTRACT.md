# House Prices Arbor continuation

Target: this isolated directory; original ml data, outputs and submission remain untouched.
Metric: all 1460-row log-price OOF RMSE, minimize. Command: `/home/xzr/bin/python evaluate.py --node-id {node_id}` on the compute server. Cheap anchor: `evaluate.py --baseline`.
Baseline: local OOF 0.10432475735803212; recorded Kaggle 0.11477.
Ambition: improve baseline with three new experiments within 30 minutes, mixed effect/novelty scope.
B_dev: KFold(5, shuffle=True, random_state=73), cached baseline predictions use this split. No pristine local B_test exists because prior model development used these labels. No pristine heldout claims permitted.
Editable surface: solution.py only. Protected: evaluate.py, config.json, protected_manifest.json, data/, baseline/. Verify SHA256 before and after each evaluation.
Candidate API: fit_predict(train_features, y_log, valid_features, test_features, foldseed) returns (valid log predictions, test log predictions), or None for unchanged baseline. Candidate never receives validation labels or SalePrice feature.
Ordinary candidate fold training excludes the two feature-defined rare rows only; all validation rows remain scored. The protected rare gate uses fold-training log-price mean, and never a test ID. Candidate blends at fixed 0.2 with baseline.
Execution: Codex keyless skill workflow, no API calls or installs. Coordinator may perform final Kaggle verification at most once; executors never access Kaggle. No PDF/report edits.

Protected legacy_features.py provides the existing Features class; fit only on candidate fold training.
