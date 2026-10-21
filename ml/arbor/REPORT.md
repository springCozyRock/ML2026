# Research Report: House Prices real keyless Arbor run:3 experiments,30minute user-authorized window.

## Results

- B_dev baseline: `0.1043`
- B_dev final trunk: `0.1043`
- B_test baseline: `0.1148`
- B_test final trunk: `0.1148`

## Exploration

- Nodes total: `3`
- Scored nodes: `3`
- Merged nodes: `0`

### Top Ideas By Score

- **3** `0.1042` _pruned_: Lasso plus pseudo-Huber XGBoost residual; robust noisy-sale loss with fixed training rounds.
- **2** `0.1053` _pruned_: Cross-fitted TargetEncoder plus category frequencies and regularized LightGBM; no validation targets in training.
- **1** `0.1058` _pruned_: Spline Ridge smooth nonlinear features; compare fixed0.2 blend without changing protected rare-sale gate.

## Global Insight

All3 experiments executed and correctly rejected. Spline and target encoding worsened development score; robust residual improved dev0.10432476 to0.10419448 but Kaggle0.11480 is worse than0.11477. Baseline preserved; independent online verification prevented accepting a false gain.

## Artifacts

- Idea tree JSON: `/Users/biantongshu/Documents/ChatGPT/Homework/ml/arbor_house_prices/.arbor/sessions/house_prices_keyless_20261013/.coordinator/idea_tree.json`
- Idea tree Markdown: `/Users/biantongshu/Documents/ChatGPT/Homework/ml/arbor_house_prices/.arbor/sessions/house_prices_keyless_20261013/.coordinator/idea_tree.md`
- Experiments: `/Users/biantongshu/Documents/ChatGPT/Homework/ml/arbor_house_prices/.arbor/sessions/house_prices_keyless_20261013/experiments`
