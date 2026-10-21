## Codebase

Working directory: /private/var/folders/78/21965t1x2w37gmpqfzcx2w_m0000gn/T/arbor-worktrees-501/codex__arbor-node-1

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 1
**Hypothesis**:
Spline Ridge smooth nonlinear features; compare fixed0.2 blend without changing protected rare-sale gate.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd /private/var/folders/78/21965t1x2w37gmpqfzcx2w_m0000gn/T/arbor-worktrees-501/codex__arbor-node-1 && /home/xzr/bin/python evaluate.py --node-id 1`
- **Dataset info**: Historical train CV used as B_dev; no pristine local B_test. Historical Kaggle score only; no new submission. Setup ended at user budget limit.
- **Baseline score**: 0.10432475735803212
- **Current trunk score**: 0.10432475735803212

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: 11 Arbor skills installed. Protected benchmark scaffold ready. Zero of three planned experiments executed; zero new Kaggle submissions. Setup exceeded requested30min; no silent extension.

## Additional Context

REAL run user-authorized anew at08:19:41 UTC. Deadline08:49:41 UTC. No package installs, no validation labels, no API/Kaggle. Use isolated SSH mirror and one fixed model, CPU8threads.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/1-<brief-description>/`.
