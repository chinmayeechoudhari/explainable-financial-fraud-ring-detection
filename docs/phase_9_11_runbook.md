# Phase 9–11 execution runbook

This branch adds the remaining research experiments without modifying the frozen
artifacts under `results/tgat/` or `results/baselines/`.

## Phase 9 — Ablations

All runs keep the chronological split, train-only scaler, seed 42, 3 epochs,
batch 256, K=10 unless the run explicitly changes K, negative ratio 20,
AdamW, learning rate 0.001, weight decay 1e-4 and best validation PR-AUC.

### Control
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/a0_baseline_control --epochs 3 --batch-size 256 --neighbor-k 10 --negative-ratio 20 --lr 0.001
```

### A — no temporal features
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/a_no_temporal_features --epochs 3 --batch-size 256 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --ablation no_temporal_features
```

### B — no historical neighbours
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/b_no_neighbors --epochs 3 --batch-size 256 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --ablation no_neighbors
```

### C — neighbour K sweep
The implementation allows `max_history` to be held at 20 while K changes,
so the sweep isolates the query neighbourhood budget more cleanly.
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/c_neighbor_k/k_1 --epochs 3 --neighbor-k 1 --max-history 20 --negative-ratio 20 --lr 0.001
python -m src.models.temporal.train_tgat --results-dir results/ablations/c_neighbor_k/k_5 --epochs 3 --neighbor-k 5 --max-history 20 --negative-ratio 20 --lr 0.001
python -m src.models.temporal.train_tgat --results-dir results/ablations/c_neighbor_k/k_10 --epochs 3 --neighbor-k 10 --max-history 20 --negative-ratio 20 --lr 0.001
python -m src.models.temporal.train_tgat --results-dir results/ablations/c_neighbor_k/k_20 --epochs 3 --neighbor-k 20 --max-history 20 --negative-ratio 20 --lr 0.001
```

### D — no time encoding
Real time gaps remain in the validity mask; only the learned time features
fed to attention key/value projections are removed.
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/d_no_time_encoding --epochs 3 --batch-size 256 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --ablation no_time_encoding
```

### E — architecture variants
```bash
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e1_one_layer --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --num-layers 1
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e2_one_head --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --num-heads 1
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e3_hidden_16 --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --hidden-dim 16
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e3_hidden_64 --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --hidden-dim 64
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e4_dropout_00 --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --dropout 0.0
python -m src.models.temporal.train_tgat --results-dir results/ablations/e_architecture/e4_dropout_04 --epochs 3 --neighbor-k 10 --negative-ratio 20 --lr 0.001 --dropout 0.4
```

Every run writes its exact configuration to `config.json`. Do not overwrite
the official TGAT result directory.

## Phase 10 — threshold selection

Run only after the official validation predictions have been verified.

```bash
python -m src.analysis.threshold_analysis --predictions results/tgat/validation_predictions.csv --test-predictions results/tgat/test_predictions.csv --results-dir results/threshold_analysis --criterion max_f1
```

The threshold is selected on validation only and persisted before test metrics
are computed. The test result is evaluated once at the frozen threshold.

## Phase 11 — future generalization

Run only after Phase 10 creates the frozen threshold.

```bash
python -m src.models.temporal.evaluate_future --data-dir data/processed/temporal_gnn --checkpoint results/tgat/model.pt --threshold-file results/threshold_analysis/frozen_threshold.json --results-dir results/generalization --neighbor-k 10 --batch-size 256
```

The checkpoint is not retrained. The scaler is fit on train only. Temporal
history is advanced train -> validation -> test -> future before future scoring.

## Verification checklist

For every Phase 9 run:
- metrics are finite;
- validation/test prediction CSVs are non-empty;
- training log exists;
- config records the intervention;
- no files are written under frozen `results/tgat/` or `results/baselines/`;
- chronological and strict-historical rules remain unchanged.

For Phase 10:
- validation has 891,571 rows and 367 positives;
- the selected threshold is explicitly marked as validation-selected;
- test is evaluated only after the threshold is frozen.

For Phase 11:
- future has 223 rows and 134 positives;
- the model checkpoint is unchanged;
- no future labels are used for tuning;
- report the 60.09% future prevalence and the small-sample limitation.
