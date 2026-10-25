"""Data loading helpers for the BT4012 Bitcoin fraud detection competition.

Every notebook loads data through this module, so file paths, column names and the
random seed are defined in exactly one place.
"""
from pathlib import Path

import pandas as pd

# Repository root = parent of src/. Using an absolute path means the notebooks
# work no matter which directory Jupyter was started from.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"              # original Kaggle CSVs (never modified)
SUBMISSIONS_DIR = ROOT / "submissions"       # CSVs uploaded to Kaggle

# Fixed seed used everywhere so results are reproducible.
SEED = 42

# Column names
ID_COL = "txId"             # unique transaction id; also used in edge list
TIME_COL = "time_step"      # coarse time snapshot (1-35 train, 36-49 test)
LABEL_COL = "label"         # 1 = illicit, 0 = licit, NaN = unknown
TEST_INDEX_COL = "index"    # row id used by submission file
FEATURE_COLS = [f"feat_{i}" for i in range(1, 166)]  # 165 anonymised, pre-normalised features


# Loaders
def load_train() -> pd.DataFrame:
    """Training transactions (time steps 1-35). `label` is NaN when unknown."""
    return pd.read_csv(RAW_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    """Test transactions (time steps 36-49). Has an extra `index` column for submission."""
    return pd.read_csv(RAW_DIR / "test.csv")


def load_edges() -> pd.DataFrame:
    """Directed payment-flow edges: an output of txId1 is spent by txId2."""
    return pd.read_csv(RAW_DIR / "txs_edgelist.csv")


def load_sample_submission() -> pd.DataFrame:
    """Kaggle's example submission: columns `index,target`, one row per test transaction."""
    return pd.read_csv(RAW_DIR / "sample_submission.csv")


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a known label (0 or 1), with the label cast to int.

    Unknown rows cannot be used to train or score a supervised model, but they are
    still useful for graph features, so the full table is kept elsewhere.
    """
    out = df[df[LABEL_COL].notna()].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def save_submission(test: pd.DataFrame, probabilities, filename: str) -> Path:
    """Write a Kaggle submission (`index,target`) to submissions/ and return its path.

    `probabilities` must be the predicted probability of illicit, in the same row order as `test`.
    """
    if len(probabilities) != len(test):
        raise ValueError(f"{len(probabilities)} probabilities for {len(test)} test rows")
    submission = pd.DataFrame({TEST_INDEX_COL: test[TEST_INDEX_COL], "target": probabilities})
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / filename
    submission.to_csv(path, index=False, lineterminator="\n")  # same bytes on Windows and Linux
    return path