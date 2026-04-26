# Final Report Artifacts

This folder consolidates the main artifacts useful for the final report on the four comparison models:

- `lstm_vanilla_20260415_163022`
- `lstm_social_20260415_163022`
- `vae_vanilla_20260415_163022`
- `vae_social_20260417_213622`

Contents:

- `summary_metrics_and_training.csv`: one-row-per-model summary with overall evaluation metrics, training duration, and key hyperparameters
- `metadata/`: copied `run_metadata.csv` files from the experiment folders
- `logs/`: trainer logs for the four main runs
- `checkpoints/pkl/`: final saved model checkpoints
- `checkpoints/state/`: final optimizer/scheduler state files
- `plots/`: copied training/validation plots
- `predictions/`: evaluation prediction folders used to compute metrics
- `scripts/`: canonical runner and reporting helpers
- `eval/Results.png`: evaluator output image present at copy time

Notes:

- The two deterministic baselines come from the `20260415_163022` run.
- The vanilla CVAE comes from the `20260415_163022` run.
- The saved social CVAE used here comes from the `20260417_213622` run.
- The social CVAE run is not a perfectly matched apples-to-apples comparison against the vanilla CVAE run:
  `batch_size=4`, `lr=1e-4`, `k=8` for social CVAE versus `batch_size=8`, `lr=1e-3`, `k=20` for vanilla CVAE.
- Prediction folders were evaluated against `DATA_BLOCK/output/test_private/`.

Metric definitions:

- `ADE`: average displacement error
- `FDE`: final displacement error
- `Col1`: predicted collision rate
- `Col2`: ground-truth collision rate in evaluator output
- `TopkADE` / `TopkFDE`: best-of-k metrics for multimodal models

