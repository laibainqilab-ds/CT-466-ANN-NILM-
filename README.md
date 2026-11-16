# CT-466 ANN-Based NILM

Course project for CT-466: Artificial Neural Network-based Non-Intrusive Load
Monitoring (NILM). The project uses a one-minute-resolution residential energy
dataset (Austin, TX) to predict individual appliance ON/OFF state and power
draw from only the whole-home aggregate power signal.

Stages 1 through 7 are complete, under a **revised, unseen-household
evaluation design**: four development households provide all training and
validation data; one household (`8386`) is held out completely and used
**only** for final test evaluation, on its full available calendar year. This
corrects the original design, in which `8386` had been trained on and
validated on exactly like every other household — which did not satisfy the
requirement that final testing use a household the model never saw during
development. This README documents what was actually built and measured
under the revised design — see
[Limitations and reproducibility notes](#20-limitations-and-reproducibility-notes)
for caveats that apply to every number below.

## Table of contents

1. [Overview](#1-project-title-and-overview)
2. [Problem statement and NILM objective](#2-problem-statement-and-nilm-objective)
3. [Dataset and selected homes](#3-dataset-and-selected-homes)
4. [Selected appliances](#4-selected-appliances)
5. [Data preprocessing and quality handling](#5-data-preprocessing-and-quality-handling)
6. [Train, validation, and test split](#6-train-validation-and-test-split)
7. [Window generation and model inputs](#7-window-generation-and-model-inputs)
8. [Ground-truth ON/OFF label generation](#8-ground-truth-onoff-label-generation)
9. [MLP architecture](#9-mlp-architecture)
10. [CNN1D architecture](#10-cnn1d-architecture)
11. [Training configurations and experiment results](#11-training-configurations-and-experiment-results)
12. [Validation-only probability threshold selection](#12-validation-only-probability-threshold-selection)
13. [Final unseen-household test evaluation](#13-final-unseen-household-test-evaluation)
14. [Actual results and CCP benchmark comparison](#14-actual-results-and-ccp-benchmark-comparison)
15. [Regression results and measurement-unit caveat](#15-regression-results-and-measurement-unit-caveat)
16. [Project directory structure](#16-project-directory-structure)
17. [Environment setup and dependencies](#17-environment-setup-and-dependencies)
18. [Execution instructions](#18-execution-instructions)
19. [Output files and plots](#19-output-files-and-plots)
20. [Limitations and reproducibility notes](#20-limitations-and-reproducibility-notes)

---

## 1. Project title and overview

**CT-466 ANN-Based NILM** trains and evaluates two neural network
architectures — a feed-forward MLP and a 1D convolutional network — that
each take a 60-minute causal window of whole-home aggregate power and
predict, per appliance, (a) whether it is ON or OFF and (b) its instantaneous
power draw. Six training configurations were run (2 architectures × 3
class-imbalance handling modes) on **development-household data only**, all
six were carried forward to validation-based probability-threshold selection,
and all six were evaluated once on the **full calendar year of the held-out,
unseen test household (`8386`)**.

## 2. Problem statement and NILM objective

Non-Intrusive Load Monitoring (NILM) aims to infer individual appliance
behavior from a single aggregate (whole-home) power meter, without
installing a dedicated sub-meter on every appliance. This project frames
NILM as a per-appliance, per-minute **multi-task** problem: a binary
classification task (is the appliance ON or OFF right now) and a regression
task (what is its power draw right now), solved jointly by one shared model
per architecture, using only the aggregate signal as input.

## 3. Dataset and selected homes

The project uses the Pecan Street / Dataport-style one-minute Austin
residential dataset, supplied as the instructor-provided archive
`1minute_data_austin.tar.gz` (project root; not committed to git — see
[§20](#20-limitations-and-reproducibility-notes)).

Five homes were approved for analysis, but they are **not** used
interchangeably. Four are **development households** (contribute training
and validation data only); one is the **official, unseen test household**
(contributes test data only, and nothing else):

| `dataid` | Role | Notes |
|---|---|---|
| 7901 | Development | Grid-only aggregate |
| 7951 | Development | Grid-only aggregate |
| 8565 | Development | Grid-only aggregate |
| 3538 | Development | Has solar generation — aggregate is `grid + solar` (null-propagating), not `grid` alone |
| **8386** | **Official unseen test household** | Grid-only aggregate. Contributes **zero** rows to training, validation, preprocessing-scaler fitting, or ground-truth threshold fitting — see §6, §8, §15 |

## 4. Selected appliances

Three sub-metered circuits are modeled, identically across all five homes:

- `air1` — central air conditioning
- `dishwasher1` — dishwasher
- `refrigerator1` — refrigerator

## 5. Data preprocessing and quality handling

**Stage 4 — cleaning and timestamp alignment** (`src/stage4_clean_align.py`):
for each home independently, `localminute` is parsed into a local naive
datetime and a UTC-normalized instant using that row's own embedded UTC
offset; chronological order is established only from these parsed values,
never from raw archive row order. Each home/split segment is reindexed onto
a complete 1-minute grid (DST-aware, via `zoneinfo`). Only the aggregate
column is interpolated, and only for gaps of 5 minutes or less with valid
data on both sides, never across a split boundary. Appliance columns are
never interpolated — each gets its own validity flag. Negative appliance
readings are diagnosed but not clipped at this stage.

Under the revised design, Stage 4 assigns **per-home** split bounds rather
than one global bound for every home (see §6): the four development homes
keep their original train/val date ranges; `8386` receives only a full-year
test range. This is a structural change in Stage 4 itself, not a filter
applied afterward.

**Stage 5 — target cleaning** (`src/stage5_clean_targets.py`): exactly two
approved transformations are applied to produce
`data/processed/five_homes_modeling.parquet`:

- **(A)** one identified extreme-anomaly row (home `7901`, train split,
  `utc_dt = 2018-09-22 01:54:00`) has `dishwasher1` and `refrigerator1` set to
  `NaN` / `valid = False`.
- **(B)** all remaining small negative appliance readings (`air1`,
  `dishwasher1`, `refrigerator1` only — never the aggregate/grid/solar
  columns) are converted to `0.0`, keeping their validity flag `True`.

The aggregate column and all other rows are passed through unchanged, and
the transformation is verified in-script (row count, schema, exactly one
invalidated row, zero remaining negatives). This stage is unaffected by the
split revision — its logic is independent of how rows are later split.

## 6. Train, validation, and test split

Splits are assigned per home by **local calendar date** (`America/Chicago`),
but the date ranges themselves are **not identical across homes** — this is
the core of the revised design (`src/stage4_clean_align.py`,
`DEV_SPLIT_BOUNDS_LOCAL` / `TEST_HOME_SPLIT_BOUNDS_LOCAL`, verified against
`data/meta/stage4_cleaning_report.json`):

| Home group | Train | Validation | Test |
|---|---|---|---|
| Development (`7901`, `7951`, `8565`, `3538`) | 2018-01-01 to 2018-09-30 | 2018-10-01 to 2018-10-31 | *(none — zero rows)* |
| **`8386` (unseen)** | *(none — zero rows)* | *(none — zero rows)* | **2018-01-01 to 2018-12-31 (full calendar year)** |

`8386` is therefore never present in any training or validation data, at any
stage of the pipeline: not in the preprocessing scaler fit (§7), not in the
ground-truth threshold fit (§8), not in model training or checkpoint
selection (§11), and not in probability-threshold selection (§12). This is
enforced structurally (Stage 4's per-home date ranges mean `8386` never
produces a train/val row in the first place) **and** defensively — Stage 6's
threshold fitting additionally raises an error if `8386`, or any unexpected
or missing household, is ever observed in the data it is given (§8).

Because splitting happens by calendar date per home before any windows are
built, and windows are later constructed independently within each
`(home, split)` segment (§7), no 60-minute window ever spans two different
splits.

## 7. Window generation and model inputs

**Stage 5B** (`src/stage5b_generate_windows.py`) builds 60-minute **causal**
windows: for each `(home, split)` pair independently, a fixed
absolute-position candidate grid is built from a stride (train stride = 5,
validation/test stride = 1) and the segment length. A candidate window is
accepted only if all 60 aggregate samples ending at that timestamp are
non-null. Each accepted window records the appliance target values and
independent per-appliance validity flags **at the single prediction
timestamp only** (never across the other 59 context samples). `(home,
split)` combinations with zero rows under the revised design (e.g. `8386` ×
train) simply produce zero accepted windows — the script does not error.

Resulting window counts (from `data/meta/stage5b_window_generation_report.json`,
this revised run):

| Split | Windows |
|---|---:|
| Train | 313,938 |
| Validation | 178,324 |
| Test | 525,422 |
| **Total** | **1,017,684** |

These counts differ from the project's original run (392,516 / 222,905 /
439,205) because `8386` no longer contributes to train/val, and now
contributes a full year instead of two months to test.

**Model input** is the 60 standardized aggregate readings only — shape
`(batch, 60)`. No home ID, timestamp, appliance reading, label, or validity
mask is ever part of the model's input tensor; everything else is carried as
separate metadata for loss computation and evaluation (`src/stage7_dataset.py`).

Preprocessing scalers (`src/stage7_preprocess.py`) are fit **strictly from
the development households' training split** — since `8386` has zero train
rows under the revised Stage 4 bounds, it structurally cannot contribute to
either scaler. Actual fitted values, from
`data/meta/stage7_preprocessing_params.json`:

| Scaler | Mean | Std | n (valid train readings) |
|---|---:|---:|---:|
| Aggregate (per-minute) | 1.269473490783557 | 1.4798533628574844 | 1,570,020 |
| `air1` target (window endpoint) | 0.4427242353585739 | 0.882397181456915 | 313,938 |
| `dishwasher1` target (window endpoint) | 0.008343325188203472 | 0.08524692032484547 | 313,889 |
| `refrigerator1` target (window endpoint) | 0.0747864722879744 | 0.07649585567025147 | 313,889 |

(`dishwasher1`/`refrigerator1` target `n` is 49 lower than `air1`'s because
of the 49 invalid train-window-endpoint rows for those two appliances — see
§8 — which are excluded from scaler fitting, not zeroed.)

## 8. Ground-truth ON/OFF label generation

**Stage 6** (`src/stage6_generate_labels.py`) assigns binary ON/OFF labels
using thresholds **fit at runtime**, then applied identically to train,
validation, and test. Under the revised design, the fitting method itself
was also corrected, for two reasons: (1) the original thresholds had been
fit using data that included the now-held-out household `8386`, and (2) the
original heuristic (Lloyd's-algorithm-style 2-means with a fixed
percentile initialization) could fail to find a true minority ("ON")
cluster smaller than its initialization percentile.

**Method — constrained exact global 1D two-means:**
for one-dimensional data, the globally optimal 2-cluster partition
minimizing total within-cluster sum of squares is always a contiguous split
of the sorted values. `fit_1d_2means_threshold()` searches every such split
point exactly (via vectorized prefix sums), rather than using a heuristic
iterative algorithm — so it is correct for any class balance, with no
initialization step to fail. An **unconstrained** version of this exact
search was found, during development, to isolate a single extreme outlier
reading as its own degenerate "cluster," producing a meaningless threshold.
To prevent this, the search is constrained so that **neither resulting
cluster may contain fewer than 0.1% of the valid readings** for that
appliance (`MIN_CLUSTER_FRACTION = 0.001`, applied symmetrically to both
sides, minimum size rounded **up**). This floor was chosen after a
sensitivity sweep across 0.01%–1.0% on the real development-household data:
results were identical and stable from 0.01% to 0.5% (a 50× range) for all
three appliances, and only degraded once the floor exceeded the smallest
observed true minority fraction (dishwasher1, 0.64%) — 0.1% sits with margin
inside that stable range. This is a documented assumption based on the three
appliances examined, not a provably universal constant.

**Population:** `fit_pooled_thresholds()` fits on **every valid, non-null,
one-minute TRAIN-split appliance reading** in
`data/processed/five_homes_modeling.parquet` — not the stride-5 window
endpoints used by the original fit — pooled across **the four development
households only**. It enforces this with an explicit whitelist *and*
exact-set check: it raises an error if any household outside the four
approved development homes is observed (rejecting `8386` specifically, by
name, among others) **and** if any of the four expected development homes
is missing — so a silently incomplete or contaminated pool is never used.

Actual regenerated thresholds (from `data/meta/stage6_thresholds.json`),
compared with the original (superseded) values:

| Appliance | Threshold (original units) | Superseded (original 5-home) value |
|---|---:|---:|
| `air1` | 1.0577275564932618 | 1.06026 |
| `dishwasher1` | 0.523375326779442 | 0.52378 |
| `refrigerator1` | 0.06521278295529205 | 0.06482 |

Rule: `label = 1` (ON) if the valid, non-null target ≥ threshold; `label = 0`
(OFF) if below threshold; `label = null` if the target itself is invalid or
missing (never silently treated as OFF).

Resulting ground-truth label counts (from
`data/meta/stage6_label_generation_report.json`):

| Split | Appliance | ON | OFF | Invalid |
|---|---|---:|---:|---:|
| Train (dev homes) | `air1` | 65,022 | 248,916 | 0 |
| Train (dev homes) | `dishwasher1` | 2,031 | 311,858 | 49 |
| Train (dev homes) | `refrigerator1` | 183,520 | 130,369 | 49 |
| Validation (dev homes) | `air1` | 20,772 | 157,552 | 0 |
| Validation (dev homes) | `dishwasher1` | 1,354 | 176,970 | 0 |
| Validation (dev homes) | `refrigerator1` | 105,498 | 72,826 | 0 |
| **Test (`8386` only, full year)** | `air1` | 43,662 | 481,760 | 0 |
| **Test (`8386` only, full year)** | `dishwasher1` | 5,212 | 520,210 | 0 |
| **Test (`8386` only, full year)** | `refrigerator1` | 285,936 | 239,486 | 0 |

(The 49 train-split invalid rows for `dishwasher1`/`refrigerator1` are the
Stage 5 anomaly-row and a small number of naturally co-missing dual-submeter
minutes surviving as accepted window endpoints — not a labeling defect.)

## 9. MLP architecture

Defined in `src/models/mlp.py`. Receives only the 60 standardized aggregate
readings — the `forward()` signature takes a single `(batch, 60)` tensor.
Architecture unchanged by the split revision.

```
Input (batch, 60)
  -> Linear(60, 128) -> ReLU -> Dropout(0.2)
  -> Linear(128, 64) -> ReLU -> Dropout(0.2)
  -> classification head: Linear(64, 3) -> Sigmoid   (3 independent probabilities)
  -> regression head:     Linear(64, 3)              (3 linear, unconstrained outputs)
```

No activation constrains the regression head: targets are standardized
(train-only mean/std) and can be negative, so an unconstrained linear output
is required.

## 10. CNN1D architecture

Defined in `src/models/cnn1d.py`. Receives the same 60 standardized
aggregate readings as the MLP, reshaped internally to a single channel.
Architecture unchanged by the split revision.

```
Input (batch, 60) -> reshape to (batch, 1, 60)
  -> Conv1d(1, 8,  kernel=5, padding=2) -> ReLU -> MaxPool1d(2)   # 60 -> 30
  -> Conv1d(8, 16, kernel=5, padding=2) -> ReLU -> MaxPool1d(2)   # 30 -> 15
  -> Conv1d(16, 32, kernel=3, padding=1) -> ReLU -> AdaptiveAvgPool1d(1)  # -> 1
  -> flatten to (batch, 32)
  -> shared dense: Linear(32, 32) -> ReLU
  -> classification head: Linear(32, 3) -> Sigmoid
  -> regression head:     Linear(32, 3)
```

Same no-Softplus rationale on the regression head as the MLP.

Both models expose **separate classification and regression heads** off a
shared trunk, and both are trained with a masked multi-task loss
(`src/stage7_losses.py`): masked BCE on the sigmoid classification outputs
plus masked MSE on the regression outputs, each **masked and averaged
independently per appliance**. Invalid/missing targets are removed via
boolean selection before any loss arithmetic runs — never zeroed out after
the fact.

## 11. Training configurations and experiment results

Six configurations were trained (`src/stage7_run_experiments.py`,
orchestrating `src/stage7_train.py`), using **development-household
training and validation data only** (`build_datasets()` reads
`five_homes_modeling.parquet` filtered to `split in ["train","val"]`, from
which `8386` is structurally absent — see §6):

- **Models:** MLP, CNN1D
- **Class-imbalance handling (BCE positive-class weighting):** `unweighted`
  (no weighting), `cap20` (weight = min(n_off/n_pos, 20) per appliance),
  `cap50` (weight = min(n_off/n_pos, 50) per appliance)

Shared hyperparameters (from each checkpoint's saved `config`):
`batch_size=256`, up to `epochs=10` with early-stopping `patience=5`,
`lr=0.001`, `lambda_cls=1.0`, `lambda_reg=1.0` (equal-weighted total loss),
`seed=42`, `device=cpu`.

Results, from `data/meta/stage7_experiment_summary.json` (this revised run;
all six completed the full 10-epoch budget — none triggered early stopping):

| Model | Class weight mode | Best epoch | Val cls loss | Val reg loss | Val total loss | Checkpoint |
|---|---|---:|---:|---:|---:|---|
| MLP | unweighted | 7 | 0.18415956455903595 | 0.5368603633501867 | 0.7210199279092226 | `models/MLP_unweighted_best.pt` |
| MLP | cap20 | 7 | 0.20646876452072613 | 0.5283620760820223 | 0.7348308406027485 | `models/MLP_cap20_best.pt` |
| MLP | cap50 | 7 | 0.23664183040568457 | 0.5325349262383657 | 0.7691767566440503 | `models/MLP_cap50_best.pt` |
| CNN1D | unweighted | 9 | 0.2007167917553866 | 0.5439785455535595 | 0.7446953373089461 | `models/CNN1D_unweighted_best.pt` |
| CNN1D | cap20 | 9 | 0.2228168148126575 | 0.5559431620901037 | 0.7787599769027612 | `models/CNN1D_cap20_best.pt` |
| CNN1D | cap50 | 9 | 0.24050234137883617 | 0.573254423750575 | 0.8137567651294112 | `models/CNN1D_cap50_best.pt` |

All six checkpoints completed this run without interruption. **All six**
were carried forward into probability-threshold selection (§12) and final
test evaluation (§13–§14) — unlike the original run, no subset was
pre-selected before evaluation.

> An earlier, now-superseded run (preserved only in the local archive — see
> §20) had an unrelated CNN1D cap50 training interruption that required a
> one-off recovery script. That incident belongs to the original,
> superseded 5-home-split results and is **not** part of this revised
> training run, which completed all six configurations normally.

## 12. Validation-only probability threshold selection

`src/stage7_evaluate.py` implements threshold selection as a strict,
structurally-enforced two-phase process. **Phase 1** (`select_validation_thresholds`)
loads a checkpoint, runs inference on the **validation split only**
(development households, October 2018), and picks one classification
probability threshold per appliance by grid search over 99 candidates (0.01
to 0.99, step 0.01), maximizing **validation Macro-F1**, with a
deterministic tie-break (the smallest threshold achieving the maximum).
This function never opens the test-split label file, so **test data (`8386`)
was not used to select these thresholds.**

These **probability thresholds** are a distinct concept from the Stage 6
**ground-truth power thresholds** in §8: Stage 6's thresholds convert raw
appliance power readings into ON/OFF ground-truth labels, once, before any
model exists. The probability thresholds below convert a trained model's own
sigmoid output into a binary ON/OFF prediction, selected per model per
appliance on validation data only, entirely independently of §8.

Frozen thresholds, all six models (from
`data/meta/stage7_validation_thresholds.json`):

| Model | `air1` | `dishwasher1` | `refrigerator1` |
|---|---:|---:|---:|
| MLP_unweighted | 0.56 | 0.11 | 0.56 |
| MLP_cap20 | 0.81 | 0.76 | 0.50 |
| MLP_cap50 | 0.79 | 0.84 | 0.47 |
| CNN1D_unweighted | 0.53 | 0.11 | 0.46 |
| CNN1D_cap20 | 0.77 | 0.72 | 0.38 |
| CNN1D_cap50 | 0.83 | 0.90 | 0.46 |

`run_full_evaluation()` enforces the ordering for real, not just by
convention: it computes Phase-1 thresholds for every checkpoint, writes them
all to disk, **re-reads that file back from disk**, and only that re-read
copy is ever passed into Phase 2 — so Phase 2 can never see an in-memory
threshold that wasn't actually persisted before any test data was touched.

## 13. Final unseen-household test evaluation

**Phase 2** (`evaluate_on_test`) takes the already-frozen thresholds as a
required argument, performs no threshold search of its own, loads the
checkpoint again, runs inference once on the **test split**, and computes
classification and regression metrics for each appliance.

Under the revised design, the test split contains **household `8386` only,
for its full available calendar year (2018-01-01 to 2018-12-31)** — the four
development households have zero rows in this split. There is therefore no
separate "pooled vs. per-home" distinction to make: the reported metrics
*are* `8386`'s metrics.

All metrics (confusion-matrix counts, Accuracy, Balanced Accuracy, Macro-F1,
MAE, RMSE) are implemented from scratch — no scikit-learn or matplotlib
dependency is used anywhere in this project. Every metric that would require
dividing by zero is reported as `null`, never fabricated as `0` — this does
not occur for any appliance in the current results, since `8386`'s full
year contains real ON and OFF samples for all three appliances.

Results are written to `data/meta/stage7_test_evaluation.json`, which also
records `"official_test_household": "8386"` explicitly.

## 14. Actual results and CCP benchmark comparison

CCP benchmarks (classification only; no MAE/RMSE threshold is specified):

| Tier | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Minimum | ≥ 0.85 | ≥ 0.75 | ≥ 0.70 |
| Higher | ≥ 0.90 | ≥ 0.85 | ≥ 0.80 |

A model/appliance combination meets a tier only if **all three** of its
Accuracy, Balanced Accuracy, and Macro-F1 meet or exceed that tier's three
thresholds simultaneously.

Test classification metrics, household `8386`, full year 2018, all six
models (from `data/meta/stage7_test_evaluation.json`; values rounded to 4
decimal places for display — the pass/fail columns are copied directly from
the report, not re-derived from the rounded figures):

| Appliance | Model | Accuracy | Balanced Accuracy | Macro-F1 | CCP minimum | CCP higher |
|---|---|---:|---:|---:|:---:|:---:|
| air1 | MLP_unweighted | 0.9859 | 0.9755 | 0.9556 | ✅ PASS | ✅ PASS |
| air1 | MLP_cap20 | 0.9858 | 0.9774 | 0.9557 | ✅ PASS | ✅ PASS |
| air1 | MLP_cap50 | 0.9847 | 0.9791 | 0.9527 | ✅ PASS | ✅ PASS |
| air1 | CNN1D_unweighted | 0.9844 | 0.9640 | 0.9503 | ✅ PASS | ✅ PASS |
| air1 | CNN1D_cap20 | 0.9851 | 0.9703 | 0.9528 | ✅ PASS | ✅ PASS |
| air1 | CNN1D_cap50 | 0.9857 | 0.9777 | 0.9552 | ✅ PASS | ✅ PASS |
| dishwasher1 | MLP_unweighted | 0.9857 | 0.7486 | 0.7032 | ❌ FAIL | ❌ FAIL |
| dishwasher1 | MLP_cap20 | 0.9855 | 0.7446 | 0.6993 | ❌ FAIL | ❌ FAIL |
| dishwasher1 | MLP_cap50 | 0.9831 | 0.7698 | 0.6921 | ❌ FAIL | ❌ FAIL |
| dishwasher1 | CNN1D_unweighted | 0.9862 | 0.7978 | 0.7294 | ✅ PASS | ❌ FAIL |
| dishwasher1 | CNN1D_cap20 | 0.9832 | 0.7671 | 0.6917 | ❌ FAIL | ❌ FAIL |
| dishwasher1 | CNN1D_cap50 | 0.9853 | 0.7714 | 0.7097 | ✅ PASS | ❌ FAIL |
| refrigerator1 | MLP_unweighted | 0.7787 | 0.7695 | 0.7719 | ❌ FAIL | ❌ FAIL |
| refrigerator1 | MLP_cap20 | 0.7749 | 0.7657 | 0.7681 | ❌ FAIL | ❌ FAIL |
| refrigerator1 | MLP_cap50 | 0.7722 | 0.7634 | 0.7657 | ❌ FAIL | ❌ FAIL |
| refrigerator1 | CNN1D_unweighted | 0.7695 | 0.7605 | 0.7627 | ❌ FAIL | ❌ FAIL |
| refrigerator1 | CNN1D_cap20 | 0.7639 | 0.7515 | 0.7534 | ❌ FAIL | ❌ FAIL |
| refrigerator1 | CNN1D_cap50 | 0.7564 | 0.7482 | 0.7501 | ❌ FAIL | ❌ FAIL |

**Honest summary, revised results:** `air1` meets **both** the minimum and
the **higher** CCP benchmark on **all six** models evaluated against the
genuinely unseen household — a stronger result than the original (same-
households) run, where no model/appliance reached the higher tier at all.
`dishwasher1` meets only the minimum tier, and only for two of six models
(`CNN1D_unweighted`, `CNN1D_cap50`); no model meets its higher tier.
`refrigerator1` **fails the minimum tier for all six models** (accuracy
0.756–0.779, short of the 0.85 floor) — consistent with the original run's
finding that refrigerator compressor cycling is a harder, lower-contrast
signal to separate from the aggregate than the other two appliances. This
project should not be described as having passed all CCP benchmarks.

## 15. Regression results and measurement-unit caveat

Test regression metrics, household `8386`, full year 2018, all six models
(from `data/meta/stage7_test_evaluation.json`; full precision, all 525,422
test windows valid for every appliance):

| Appliance | Model | MAE | RMSE |
|---|---|---:|---:|
| air1 | MLP_unweighted | 0.07526586204767227 | 0.214969202876091 |
| air1 | MLP_cap20 | 0.0785602554678917 | 0.2146041840314865 |
| air1 | MLP_cap50 | 0.0821872279047966 | 0.21775975823402405 |
| air1 | CNN1D_unweighted | 0.07434840500354767 | 0.21933938562870026 |
| air1 | CNN1D_cap20 | 0.07707718014717102 | 0.21825411915779114 |
| air1 | CNN1D_cap50 | 0.08659190684556961 | 0.21492473781108856 |
| dishwasher1 | MLP_unweighted | 0.019437480717897415 | 0.09933723509311676 |
| dishwasher1 | MLP_cap20 | 0.020912744104862213 | 0.09966496378183365 |
| dishwasher1 | MLP_cap50 | 0.021393463015556335 | 0.09970821440219879 |
| dishwasher1 | CNN1D_unweighted | 0.024586809799075127 | 0.09873536229133606 |
| dishwasher1 | CNN1D_cap20 | 0.021787233650684357 | 0.10165110975503922 |
| dishwasher1 | CNN1D_cap50 | 0.025105755776166916 | 0.10189829021692276 |
| refrigerator1 | MLP_unweighted | 0.04370512068271637 | 0.059966277331113815 |
| refrigerator1 | MLP_cap20 | 0.046253006905317307 | 0.06084933504462242 |
| refrigerator1 | MLP_cap50 | 0.045939911156892776 | 0.06067792326211929 |
| refrigerator1 | CNN1D_unweighted | 0.045380476862192154 | 0.06171959266066551 |
| refrigerator1 | CNN1D_cap20 | 0.04690278321504593 | 0.06255471706390381 |
| refrigerator1 | CNN1D_cap50 | 0.04686581715941429 | 0.06280366331338882 |

**Unit caveat (unchanged by the split revision):** these values are in the
dataset's **original, unconverted measurement units**. The physical unit of
the source appliance and aggregate readings (watts vs. kilowatts) could not
be independently verified from any documentation available in this project
— the archive contains only a directory entry, the main CSV, and
`metadata.csv`, and no unit string was found in any of them
(`data/meta/stage6_thresholds.json`, `data/meta/stage7_preprocessing_params.json`).
No unit conversion or assumption is made anywhere in this project; MAE/RMSE
above should **not** be labeled "W" or "kW" without separately confirming
the unit against dataset documentation outside this repository.

## 16. Project directory structure

```
ANN_NILM/
├── README.md
├── .gitignore
├── 1minute_data_austin.tar.gz        # instructor-provided archive (git-ignored)
├── recover_cnn1d_cap50.py            # historical one-off script, tied to the superseded
│                                      # original-run recovery incident (§11) — not part of
│                                      # this revised pipeline
├── src/
│   ├── dry_run_single_home.py        # Stage 2 — single-home extraction dry run
│   ├── extract_five_homes.py         # Stage 3 — full five-home extraction
│   ├── stage4_clean_align.py         # Stage 4 — cleaning + alignment + PER-HOME split assignment
│   ├── stage5_clean_targets.py       # Stage 5 — target (appliance reading) cleaning
│   ├── stage5b_generate_windows.py   # Stage 5B — 60-minute causal window generation
│   ├── stage6_generate_labels.py     # Stage 6 — constrained exact 2-means thresholds + ON/OFF labels
│   ├── stage7_preprocess.py          # Stage 7 — dev-homes-only train scaler fitting + persistence
│   ├── stage7_dataset.py             # Stage 7 — NILMWindowDataset / collate_fn
│   ├── stage7_losses.py              # Stage 7 — masked multi-task losses, pos-weight calc
│   ├── stage7_step4a_prototype.py    # Stage 7 — small-subset training prototype
│   ├── stage7_train.py               # Stage 7 — TrainConfig, run_epoch, train_model
│   ├── stage7_run_experiments.py     # Stage 7 — six-experiment orchestrator
│   ├── stage7_evaluate.py            # Stage 7 — validation threshold selection + 8386-only test evaluation
│   └── models/
│       ├── mlp.py                    # MLP architecture
│       └── cnn1d.py                  # 1D-CNN architecture
├── tests/                            # unittest suite: test_stage6_thresholds.py,
│                                      # test_stage7_{dataset,evaluate,losses,models,
│                                      # run_experiments,step4a_prototype,training}.py
├── data/
│   ├── interim/                      # Stage 2/3 intermediate output (git-ignored)
│   ├── processed/                    # Stage 4-6 parquet outputs (git-ignored)
│   └── meta/
│       ├── stage4_cleaning_report.json
│       ├── stage5_target_cleaning_report.json
│       ├── stage5b_window_generation_report.json
│       ├── stage6_thresholds.json
│       ├── stage6_label_generation_report.json
│       ├── stage7_preprocessing_params.json
│       ├── stage7_experiment_summary.json
│       ├── stage7_validation_thresholds.json
│       ├── stage7_test_evaluation.json
│       └── stage7_evaluation_plots/
│           └── {MLP,CNN1D}_{unweighted,cap20,cap50}/{air1,dishwasher1,refrigerator1}.svg
│               # six model folders, three SVGs each = 18 plots, all plotting household 8386
├── models/                           # the six revised-run checkpoints (git-ignored)
└── archive/                           # LOCAL, git-ignored historical snapshot of the ORIGINAL
                                       # (pre-revision) 5-home-split run -- data, code, checkpoints,
                                       # and results. Not part of the committed GitHub repository.
```

(`models_recovery/CNN1D_cap50_best.pt`, git-ignored, still exists on disk as
a leftover from the *original* run's recovery incident — see §11, §20. It is
not produced or needed by this revised run.)

## 17. Environment setup and dependencies

Observed working environment (this project's existing `.venv`; **not
re-verified as part of this revision** — unchanged from the prior audit):

- **Python 3.12** (3.12.10)
- **torch** 2.15.0+cpu
- **pandas** 3.1.1
- **numpy** 2.5.4
- **pyarrow** 26.0.1

No `requirements.txt` currently exists in this repository. The four
third-party packages above (plus the Python standard library) are the only
dependencies imported anywhere in `src/` or `tests/` — there is **no
scikit-learn and no matplotlib dependency**: all evaluation metrics are
hand-implemented in `stage7_evaluate.py`, and all plots are hand-written SVG
rather than rendered by a plotting library.

PowerShell setup:

```powershell
# From the project root (E:\CT-466\ANN_NILM)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch pandas numpy pyarrow
```

## 18. Execution instructions

Scripts fall into three categories: **data-generating** (Stages 1–6, produce
intermediate/processed files), **training** (`stage7_run_experiments.py`,
produces checkpoints), and **evaluation** (`stage7_evaluate.py`, reads
checkpoints only and never modifies training code or labels). Most scripts
are plain top-level scripts with no CLI arguments — run them directly with
`python`.

**Stage 1 — dataset acquisition.** No script. Place the instructor-provided
`1minute_data_austin.tar.gz` archive at the project root.

**Stages 2–6 — data generation**, run in order (each depends on the
previous stage's output file(s)):

```powershell
python src\dry_run_single_home.py      # Stage 2: single-home dry run -> data\interim\home_7951_dry_run.csv
python src\extract_five_homes.py       # Stage 3: full extraction      -> data\interim\five_homes.parquet
python src\stage4_clean_align.py       # Stage 4: cleaning + PER-HOME split assignment -> data\processed\five_homes_aligned.parquet
python src\stage5_clean_targets.py     # Stage 5: target cleaning      -> data\processed\five_homes_modeling.parquet
python src\stage5b_generate_windows.py # Stage 5B: window generation   -> data\processed\window_index_{train,val,test}.parquet
python src\stage6_generate_labels.py   # Stage 6: fits thresholds (dev homes only) + ON/OFF labels -> data\processed\window_labels_{train,val,test}.parquet
```

> `stage5b_generate_windows.py` and `stage6_generate_labels.py` each refuse
> to run (raise `RuntimeError`) if their own output files already exist, to
> prevent accidental overwrites — move or rename the existing output files
> first (e.g. into a local archive) if you genuinely intend to regenerate
> them.
>
> Stages 2–5B use **hardcoded absolute Windows paths** rooted at
> `E:\CT-466\ANN_NILM`; they will only work unmodified if the project
> actually lives at that exact path. Stage 6 onward use
> `Path(__file__).resolve().parent.parent` and are portable.

**Stage 7 — preprocessing metadata** (fits scalers on dev-homes-only train
data and persists them):

```powershell
python src\stage7_preprocess.py
```

> ⚠️ This overwrites `data/meta/stage7_preprocessing_params.json`
> unconditionally.

**Stage 7 — training** (produces the six checkpoints under `models/`):

```powershell
python src\stage7_run_experiments.py
```

> ⚠️ This overwrites any existing checkpoint files of the same name under
> `models/` and rewrites `data/meta/stage7_experiment_summary.json`
> unconditionally. The results currently documented in this README (§11)
> already reflect the most recent such run under the revised split; running
> this again will replace them with a new training run's results.

**Stage 7 — evaluation** (reads saved checkpoints only; does not train):

```powershell
python src\stage7_evaluate.py
```

> ⚠️ The script's own `__main__` block evaluates only two checkpoints by
> default (`MLP_unweighted`, `CNN1D_unweighted`); the full six-model results
> documented in this README (§12–§15) were produced by calling
> `run_full_evaluation()` directly with all six checkpoint paths. Either
> way, this unconditionally overwrites
> `data/meta/stage7_validation_thresholds.json`,
> `data/meta/stage7_test_evaluation.json`, and every SVG under
> `data/meta/stage7_evaluation_plots/`.
>
> `recover_cnn1d_cap50.py` is a historical one-off script tied to the
> *original* (superseded) run's recovery incident (§11); it is not part of
> this pipeline and is not expected to be needed again.

**Running the test suite:**

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

> The test suite was not re-run in full as part of this revision, so no
> pass/fail count is reported here — the prior README's "163 tests, 163
> passed" figure predates this revision's additions to the suite and should
> not be treated as current. Run the command above to obtain a verified,
> up-to-date result.

## 19. Output files and plots

| File | Produced by | Contents |
|---|---|---|
| `data/meta/stage4_cleaning_report.json` | Stage 4 | Per-home cleaning/alignment diagnostics, now reflecting per-home split bounds |
| `data/meta/stage5_target_cleaning_report.json` | Stage 5 | Target-cleaning verification report |
| `data/meta/stage5b_window_generation_report.json` | Stage 5B | Per-home/split window counts under the revised split |
| `data/meta/stage6_thresholds.json` | Stage 6 | Re-derived ON/OFF thresholds, fitting method/provenance, and the superseded original values |
| `data/meta/stage6_label_generation_report.json` | Stage 6 | Per-split/appliance ON/OFF/invalid counts under the revised split |
| `data/meta/stage7_preprocessing_params.json` | `stage7_preprocess.py` | Dev-homes-only, train-split-only aggregate/target scaler statistics, with explicit source-split/source-homes provenance |
| `data/meta/stage7_experiment_summary.json` | `stage7_run_experiments.py` | All six experiments' status, best epoch, losses, checkpoint paths |
| `data/meta/stage7_validation_thresholds.json` | `stage7_evaluate.py` (Phase 1) | Frozen per-appliance probability thresholds for all six models |
| `data/meta/stage7_test_evaluation.json` | `stage7_evaluate.py` (Phase 2) | Classification and regression metrics, CCP pass/fail, for all six models, on household `8386`'s full year only |
| `data/meta/stage7_evaluation_plots/{MODEL}/{appliance}.svg` | `stage7_evaluate.py` (Phase 2) | Actual-vs-predicted power plots, household `8386`, first ~48 hours of its test year, identical interval across all six models |
| `models/*.pt` | `stage7_run_experiments.py` | The six revised-run training checkpoints |

## 20. Limitations and reproducibility notes

- **Measurement unit not verified.** All power values (aggregate and
  per-appliance) are reported in the dataset's original, unconverted units.
  Whether that unit is watts or kilowatts could not be confirmed from any
  documentation shipped with the dataset archive — do not label results "W"
  or "kW" without independently verifying this against outside
  documentation.
- **Portability.** Stages 2–5B use hardcoded absolute paths rooted at
  `E:\CT-466\ANN_NILM`. Running this project from a different location on a
  different machine requires either placing the project at that exact path
  or editing those path constants.
- **Dataset and intermediate data are not in version control.** The raw
  archive, `data/interim/`, and `data/processed/` are all git-ignored. This
  repository does not let you reproduce Stages 1–6 output without separately
  obtaining the instructor-provided dataset archive.
- **Local historical archive.** `archive/` is a **local, git-ignored**
  directory preserving the complete original (pre-revision) 5-home-split
  project snapshot — data, source code, checkpoints, and results — for
  historical comparison. It is **not** part of the committed GitHub
  repository and will not be present in a fresh clone.
- **Original CNN1D cap50 recovery incident (historical, superseded).** In
  the *original* 5-home-split run, that training configuration was
  interrupted partway through and required a one-off recovery script
  (`recover_cnn1d_cap50.py`) into a separate `models_recovery/` checkpoint.
  That incident is preserved in `archive/` for the record; it is **not**
  part of the current revised run, whose CNN1D cap50 configuration
  completed normally (§11).
- **Threshold re-derivation changes downstream numbers.** Re-fitting the
  ON/OFF thresholds from development-homes-only data (§8) produced
  different threshold values than the original run, and therefore different
  label counts, training losses, and final CCP outcomes than the original
  (same-households) run. This is intentional, not a regression.
- **Partial, not full, CCP success.** `air1` meets both CCP tiers on all
  six models against the unseen household; `dishwasher1` meets only the
  minimum tier, and only for two of six models; `refrigerator1` fails the
  minimum tier for all six. See §14 for the full breakdown.
- **Re-running Stage 7 scripts overwrites existing results.**
  `stage7_preprocess.py`, `stage7_run_experiments.py`, and
  `stage7_evaluate.py` all overwrite their output files unconditionally on
  every run; see the warnings in §18 before invoking any of them.
- **Test-suite status not currently verified.** See the note in §18 — no
  pass/fail count is claimed in this revision.
- **No scikit-learn, no matplotlib.** All classification/regression metrics
  and all evaluation plots are implemented from scratch in
  `stage7_evaluate.py`, by design, not because of an oversight.
