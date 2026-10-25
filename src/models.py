"""Model definitions shared by all notebooks.

- make_logreg / make_random_forest / make_lightgbm return a NEW, untrained model. 
  The validation code calls them once per fold, so no fold can reuse what another fold has learned.
- fit_lightgbm / two_stage_model / average_over_seeds: the graph model of 04_graph_features and
  the final model of 05_model_comparison

A "model function" below is a function model(train_rows, pred_rows, seed) that trains on 
train_rows and returns the probability of illicit for each row of pred_rows.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS, LABEL_COL, SEED, TIME_COL
from src.features import GRAPH_COLS, Graph, neighbour_mean_max

# LightGBM threads, kept fixed: LightGBM's sums can differ in the last digits
# when the number of threads changes, and 2 is the setting the leaderboard files were made with.
N_THREADS = 2


def make_logreg(balanced: bool = False):
    """Logistic regression: a simple linear baseline. 
    Features are standardised first because this model is sensitive to feature scale."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None),
    )


def make_random_forest(balanced: bool = False):
    """Random forest: many decision trees trained on random subsets of rows and features,
    with their votes averaged."""
    return RandomForestClassifier(
        n_estimators=300,
        # "balanced_subsample" re-weights classes inside each tree's bootstrap sample
        class_weight="balanced_subsample" if balanced else None,
        n_jobs=-1,              # use all CPU cores
        random_state=SEED,
    )


def make_lightgbm(balanced: bool = False, seed: int = SEED):
    """LightGBM: gradient-boosted trees, 
    where each new tree corrects the errors of the previous ones.
    `seed` changes the random row and feature sampling (used to average seeds later)."""
    return lgb.LGBMClassifier(
        n_estimators=500,       # number of trees
        learning_rate=0.03,     # small steps: more trees, but less overfitting
        num_leaves=31,          # maximum complexity of each tree
        subsample=0.8,          # each tree sees a random 80% of rows ...
        subsample_freq=1,       # ... re-drawn for every tree
        colsample_bytree=0.5,   # ... and a random 50% of features (helps with correlated features)
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=N_THREADS,
        verbose=-1,             # silence training logs
    )


def logit(p):
    """Log-odds of a probability."""
    return np.log(p / (1 - p))


def step_offset(rows: pd.DataFrame) -> np.ndarray:
    """LightGBM starting score for each row: logit(its step's illicit rate) - logit(the overall rate).
    
    Explained and used in 05_model_comparison. 
    Uses only the labels of `rows`, which are always training rows.
    The rates are smoothed as (illicit rows + 0.5) / (rows + 1),
    so the log-odds stay finite for a step with no illicit rows.
    """
    y = rows[LABEL_COL]
    per_step = y.groupby(rows[TIME_COL]).agg(["sum", "size"])
    step_rate = (per_step["sum"] + 0.5) / (per_step["size"] + 1)
    overall_rate = (y.sum() + 0.5) / (len(y) + 1)
    return (logit(step_rate) - logit(overall_rate)).reindex(rows[TIME_COL]).to_numpy()


def fit_lightgbm(rows: pd.DataFrame, cols: list, seed: int, balanced: bool = False, offset=None):
    """Fit make_lightgbm on rows[cols]. `offset` (one value per row, or None) is the fixed starting score."""
    # Predictions use the trees only: LightGBM does not add init_score when predicting
    return make_lightgbm(balanced, seed).fit(rows[cols], rows[LABEL_COL], init_score=offset)


RISK_COLS = ["nb_mean_risk", "nb_max_risk"]
STAGE2_COLS = FEATURE_COLS + GRAPH_COLS + RISK_COLS  # 165 raw + 27 graph + 2 risk = 194 columns


def add_neighbour_risk(graph: Graph, train_rows: pd.DataFrame, pred_rows: pd.DataFrame, seed: int,
                       balanced: bool = False, use_offset: bool = True) -> list:
    """Stage 1: return copies of train_rows and pred_rows with the 2 neighbour-risk columns (RISK_COLS) appended."""
    offset = step_offset(train_rows) if use_offset else None

    # Out-of-fold scores for the training rows: 5 groups of whole time steps. 
    # A transaction's present neighbours are in its own time step, 
    # so each neighbour is scored by a model that never saw its step.
    oof = np.zeros(len(train_rows))
    for fit_idx, score_idx in GroupKFold(n_splits=5).split(train_rows, groups=train_rows[TIME_COL]):
        part_offset = offset[fit_idx] if use_offset else None
        model = fit_lightgbm(train_rows.iloc[fit_idx], FEATURE_COLS, seed, balanced, part_offset)
        oof[score_idx] = model.predict(train_rows.iloc[score_idx][FEATURE_COLS], raw_score=True)

    # Prediction rows: a stage-1 model trained on all training rows
    full = fit_lightgbm(train_rows, FEATURE_COLS, seed, balanced, offset)
    pred_score = full.predict(pred_rows[FEATURE_COLS], raw_score=True)

    # Mean and max score over the present neighbours (present = training rows + prediction rows)
    present = pd.concat([train_rows, pred_rows])
    scores = np.concatenate([oof, pred_score])[:, None]
    return [pd.concat([rows, neighbour_mean_max(graph, rows, present, scores, ["risk"])], axis=1)
            for rows in (train_rows, pred_rows)]


def two_stage_model(graph: Graph, train_rows: pd.DataFrame, pred_rows: pd.DataFrame, seed: int,
                    balanced: bool = False, use_offset: bool = True) -> np.ndarray:
    """Stage 1 -> neighbours' risk -> stage 2. Returns the stage-2 probability of each prediction row.

    train_rows and pred_rows must already have the 27 graph columns (add_graph_features).
    The labels of pred_rows are never read.
    balanced=True, use_offset=False is the model of 04_graph_features (public 0.95846);
    the defaults are the final model of 05_model_comparison (public 0.96169).
    """
    train_rows, pred_rows = add_neighbour_risk(graph, train_rows, pred_rows, seed, balanced, use_offset)
    offset = step_offset(train_rows) if use_offset else None
    model = fit_lightgbm(train_rows, STAGE2_COLS, seed, balanced, offset)
    return model.predict_proba(pred_rows[STAGE2_COLS])[:, 1]  # trees only: init_score is not added


def average_over_seeds(model, train_rows: pd.DataFrame, pred_rows: pd.DataFrame, seeds: list) -> np.ndarray:
    """Mean predicted probability of the model function `model(train_rows, pred_rows, seed)` over `seeds`."""
    return np.mean([model(train_rows, pred_rows, seed) for seed in seeds], axis=0)