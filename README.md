# BT4012 Bitcoin Transaction Fraud Detection

This repository contains the notebooks and code for the BT4012 Kaggle
competition on classifying Bitcoin transactions as licit (0) or illicit (1).
The evaluation metric is ROC-AUC. The test transactions come from later time
steps (36–49) than the labeled training transactions (1–35), so validation must
respect time order.

## Final submissions

**`e1156682.ipynb`** (repository root) is the notebook for my top submission
(public leaderboard **0.96169**). It is self-contained (it does not import from
`src/`) and covers data preprocessing, feature engineering, temporal validation
and generating the submission CSVs. Run it top to bottom from the repository
root after the data and environment setup below; it takes about 10 minutes.

It regenerates **both** files I selected for the private leaderboard:

| File written | Model | Submitted on Kaggle as | Public LB |
|---|---|---|---|
| `submissions/e1156682_submission.csv` | Final model (notebook section 7.1) | `probe_nocw_stepoff.csv` | **0.96169** |
| `submissions/e1156682_submission_2.csv` | Neighbours'-risk model (notebook section 7.2) | `probe_neighbour_risk.csv` | 0.95846 |

The final model is a two-stage (stacked) LightGBM:

1. **Graph features.** 27 features describe each transaction's neighbourhood
   in the payment graph: degree and presence counts, plus the mean and max of
   10 principal components of the neighbours' local features. A neighbour counts
   only if it is **present** (a labeled training row or a `test.csv` row), so
   training rows see their neighbourhood the same way test rows do.
2. **Stage 1** scores every present transaction from its 165 raw features.
   Training rows get out-of-fold scores (5 groups of whole time steps), so no
   neighbour is scored by a model that saw its label.
3. **Stage 2** adds the mean and max stage-1 score of each transaction's present
   neighbours (illicit transactions cluster together) and trains on all 194
   columns.
4. Both stages use **no class weight** and a **per-step offset**: each training row
   starts from its time step's illicit log-odds, so the trees learn what
   separates illicit from licit within a step instead of each step's base rate.
   The prediction is averaged over 6 seeds.

The second file is the same two-stage model with the balanced class weight and
no offset, averaged over 3 seeds. It does not depend on the last step, which
was chosen on the public leaderboard.

| Submission | Public LB |
|---|---|
| Random forest (balanced), 165 raw features | 0.93650 |
| LightGBM (balanced), 165 raw features | 0.94637 |
| + 27 present-neighbour graph features | 0.95121 |
| + neighbours' predicted risk (two-stage stacking) | 0.95846 |
| **+ no class weight and per-step offset (final)** | **0.96169** |

## Data setup

Download the competition data files from the Kaggle competition page after
joining the competition. Place the files directly in `data/raw/` (relative to
the repository root) using these names:

- `train.csv`
- `test.csv`
- `txs_edgelist.csv`
- `sample_submission.csv`

The `data/` directory is ignored by Git and must not be committed. The
notebooks and source code expect the paths above.

## Environment

From the repository root, create and activate a virtual environment, then
install the dependencies (tested with Python 3.14.6):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`.

`e1156682.ipynb` reads `data/raw/` relative to the folder it runs in, so start
Jupyter from the repository root. The notebooks in `notebooks/` resolve paths
in `src/data.py` relative to the repository root, so they can be launched from
any directory.

## Project structure

```
data/raw/            Kaggle files (not in Git)
e1156682.ipynb       Final model notebook (both final submissions, public LB 0.96169 and 0.95846)
notebooks/           Development history, read in order:
                       01_eda               data checks, drift over time, homophily, the present-neighbour rule
                       02_validation_setup  temporal validation design, the step-24 event, late folds
                       03_baseline_models   LR / RF / LightGBM baselines (LB 0.94637)
                       04_graph_features    graph features and the two-stage model (LB 0.95121, 0.95846)
                       05_model_comparison  class weight x per-step offset, all 21 submissions, final picks (LB 0.96169)
src/                 data.py (paths, loaders), features.py (graph features), models.py (models, two-stage model),
                     validation.py (time-based folds, experiment log)
submissions/         Submission CSVs (not in Git)
experiments.csv      Validation AUC of every model run in notebooks 03-05
requirements.txt     Pinned package versions
```

## Reproducibility

- Validation is temporal only: train on earlier time steps, validate on later ones.
  Models are chosen on four late folds (validate on steps 25–35, 27–35, 29–35, 31–35).
- `time_step` is never a model feature. It is only used for validation splits,
  the out-of-fold groups in stage 1 and the per-step offset, which LightGBM does
  not add at prediction time.
- Random seeds are fixed (42 for validation runs; 42–47 for the final model;
  42–44 for the second submission), and LightGBM uses a fixed 2 threads, because
  its sums can change in the last digits with the thread count.
- `e1156682.ipynb` reproduces both submitted files exactly. Notebooks 04 and 05
  build the same two files from `src/` (`04_neighbour_risk.csv`, `05_final_model.csv`).