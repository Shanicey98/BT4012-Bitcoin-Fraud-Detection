"""Time-based validation helpers.

The test set comes from later time steps (36-49) than the training data (1-35), and the
illicit share drifts over time. To estimate test performance honestly, every model is
trained on earlier time steps and scored on the time steps that follow
("expanding-window" validation), never on randomly shuffled rows.
"""
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import LABEL_COL, ROOT, SEED, TIME_COL, labeled
from src.features import Graph, add_graph_features, fit_local_pca

# Each fold: (last training time step, (first, last) validation time step).
# The training window grows from fold to fold, like the final model, which trains on
# everything up to step 35 and predicts the steps that follow."
FOLDS = [
    (20, (21, 25)),
    (25, (26, 30)),
    (30, (31, 35)),
]

# "Late" folds: every training window contains steps 21-24, where a new kind of illicit
# transaction appears (02_validation_setup, section 4), and each validates on all later steps.
# These are the folds used to choose between models.
LATE_FOLDS = [
    (24, (25, 35)),
    (26, (27, 35)),
    (28, (29, 35)),
    (30, (31, 35)),
]

EXPERIMENTS_CSV = ROOT / "experiments.csv"  # one row per evaluated model, for comparison and the report


def time_splits(df: pd.DataFrame, folds=FOLDS):
    """Yield (fold number, training rows, validation rows) for each time-based fold."""
    for i, (train_end, (val_start, val_end)) in enumerate(folds, start=1):
        train_part = df[df[TIME_COL] <= train_end]
        valid_part = df[df[TIME_COL].between(val_start, val_end)]
        yield i, train_part, valid_part


def evaluate(make_model, df: pd.DataFrame, features: list, folds=FOLDS) -> pd.DataFrame:
    """Train a fresh model on each fold and return one row of results per fold.

    `make_model` is a function that returns a new, untrained model, so no fold can
    reuse what another fold has learned.
    """
    rows = []
    for i, train_part, valid_part in time_splits(df, folds):
        model = make_model()
        model.fit(train_part[features], train_part[LABEL_COL])
        pred = model.predict_proba(valid_part[features])[:, 1]  # probability of illicit
        rows.append({
            "fold": i,
            "train_steps": f"1-{train_part[TIME_COL].max()}",
            "valid_steps": f"{valid_part[TIME_COL].min()}-{valid_part[TIME_COL].max()}",
            "n_valid": len(valid_part),
            "n_illicit": int(valid_part[LABEL_COL].sum()),
            "auc": roc_auc_score(valid_part[LABEL_COL], pred),
        })
    return pd.DataFrame(rows)


def evaluate_graph_model(model, train: pd.DataFrame, graph: Graph, folds=LATE_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Like `evaluate`, but for a model function that needs the graph features rebuilt inside each fold.

    `model(train_rows, pred_rows, seed)` must return the probability of illicit 
    for each row of pred_rows (see src/models.py).
    `train` is the full train.csv table (labeled and unlabeled rows).

    Each fold rebuilds everything from its own training period, exactly like the submission:
    the PCA is fitted on the train.csv rows up to the fold's last training step, 
    the present set is the fold's training rows plus its validation rows, 
    and the validation labels are removed before any feature is built 
    (they are used only to compute the AUC).
    """
    lab = labeled(train).reset_index(drop=True)
    rows = []
    for i, (train_end, (val_start, val_end)) in enumerate(folds, start=1):
        fit_rows = lab[lab[TIME_COL] <= train_end]
        valid_rows = lab[lab[TIME_COL].between(val_start, val_end)]
        valid_labels = valid_rows[LABEL_COL]
        valid_rows = valid_rows.drop(columns=LABEL_COL)  # from here on, no model or feature can see them
        pca = fit_local_pca(train[train[TIME_COL] <= train_end])  # PCA from the fold's training period only
        fit_g, valid_g = add_graph_features(graph, fit_rows, valid_rows, pca)
        pred = model(fit_g, valid_g, seed)
        rows.append({
            "fold": i,
            "train_steps": f"1-{train_end}",
            "valid_steps": f"{val_start}-{val_end}",
            "n_valid": len(valid_rows),
            "n_illicit": int(valid_labels.sum()),
            "auc": roc_auc_score(valid_labels, pred),
        })
    return pd.DataFrame(rows)


def log_experiment(name: str, results: pd.DataFrame, notes: str = "") -> None:
    """Save one model's fold AUCs in experiments.csv so all runs can be compared later.

    If the same model was already logged on the same folds, its old row is replaced,
    so re-running a notebook does not create duplicate rows.
    """
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": name,
        "folds": " | ".join(results["valid_steps"]),  # which validation steps were used
        "mean_auc": round(results["auc"].mean(), 4),
        "worst_fold_auc": round(results["auc"].min(), 4),  # how badly it can fail under drift
        **{f"fold{f}_auc": round(a, 4) for f, a in zip(results["fold"], results["auc"])},
        "notes": notes,
    }

    log = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        old = pd.read_csv(EXPERIMENTS_CSV)
        same_run = (old["model"] == name) & (old["folds"] == row["folds"])
        # Re-read and re-write the whole file, so runs with 3 folds and with 4 folds still line up
        log = pd.concat([old[~same_run], log], ignore_index=True)
    log = log[[c for c in log.columns if c != "notes"] + ["notes"]]  # keep notes as the last column
    log.to_csv(EXPERIMENTS_CSV, index=False)