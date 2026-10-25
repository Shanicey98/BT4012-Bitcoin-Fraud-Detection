"""Graph features built from the payment-flow edge list.

An edge txId1 -> txId2 means an output of txId1 is spent by txId2. 
Every edge between two transactions that have a row stays inside one time step, 
and no edge links a training transaction to a test transaction.

Key idea: "present" neighbours.
test.csv seems to contain only the LABELED transactions of steps 36-49. 
Unlabeled test-period transactions still appear in the edge list, but they have no features. 
So a test transaction can only "see" some of its neighbours.

To make training rows look like test rows, a neighbour counts as PRESENT only if it is
    - a labeled row of train.csv (for training transactions), or
    - a row of test.csv (for test transactions).

`add_graph_features` applies this one rule to training and prediction rows in a single call:
present = the training rows plus the prediction rows. 

No label values are used: the only label information is which training rows are labeled, 
and test.csv carries the same information for the test period.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS, ID_COL

# The first 93 features describe the transaction itself ("local" features in the original Elliptic data);
# most of the rest are neighbour summaries made by the data provider.
LOCAL_COLS = FEATURE_COLS[:93]
N_COMPONENTS = 10  # local features are compressed to 10 principal components before averaging

PRESENCE_COLS = ["in_degree", "out_degree", "n_in_present", "n_out_present",
                 "n_nb_present", "share_present", "n_nb_absent"]
PC_COLS = [f"pc{i}" for i in range(N_COMPONENTS)]
GRAPH_COLS = PRESENCE_COLS + [f"nb_mean_{c}" for c in PC_COLS] + [f"nb_max_{c}" for c in PC_COLS]  # 27 columns


class Graph(NamedTuple):
    """The payment graph as sparse 0/1 matrices, so "sum over my neighbours" is a fast matrix product."""
    nodes: pd.Index      # transaction id of each node number 0..n-1
    out_adj: csr_matrix  # out_adj[i, j] = 1: transaction j spends an output of i
    in_adj: csr_matrix   # in_adj[i, j] = 1: transaction j pays into i
    nb_adj: csr_matrix   # neighbours in either direction, each counted once


def build_graph(edges: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> Graph:
    """Number every transaction (edge list, train.csv and test.csv) 0..n-1 and build the Graph."""
    nodes = pd.Index(pd.unique(pd.concat([edges["txId1"], edges["txId2"], train[ID_COL], test[ID_COL]])))
    src, dst = nodes.get_indexer(edges["txId1"]), nodes.get_indexer(edges["txId2"])
    out_adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(nodes), len(nodes))).tocsr()
    out_adj.data[:] = 1          # a repeated edge counts once
    in_adj = out_adj.T.tocsr()
    nb_adj = out_adj + in_adj
    nb_adj.data[:] = 1           # a neighbour in both directions counts once
    return Graph(nodes, out_adj, in_adj, nb_adj)


def presence_features(graph: Graph, rows: pd.DataFrame, present_ids) -> pd.DataFrame:
    """Degree and neighbour-presence counts for each transaction in `rows` (same row order)."""
    present = graph.nodes.isin(present_ids).astype(float)  # 1 = present (the rule above)
    ix = graph.nodes.get_indexer(rows[ID_COL])
    degree = np.asarray(graph.nb_adj.sum(axis=1)).ravel()[ix]
    n_present = (graph.nb_adj @ present)[ix]
    return pd.DataFrame({
        "in_degree": np.asarray(graph.out_adj.sum(axis=0)).ravel()[ix],   # transactions paying into it
        "out_degree": np.asarray(graph.out_adj.sum(axis=1)).ravel()[ix],  # transactions spending its outputs
        "n_in_present": (graph.in_adj @ present)[ix],
        "n_out_present": (graph.out_adj @ present)[ix],
        "n_nb_present": n_present,
        "share_present": np.where(degree > 0, n_present / np.maximum(degree, 1), -1.0),  # -1 = no neighbours
        "n_nb_absent": degree - n_present,
    }, index=rows.index)


def neighbour_mean_max(graph: Graph, rows: pd.DataFrame, present: pd.DataFrame, values, names) -> pd.DataFrame:
    """Mean and max of `values` over the PRESENT neighbours of each transaction in `rows`.

    `values` has one row per transaction in `present` (same order) and one column per name in `names`.
    Transactions with no present neighbour get NaN, which LightGBM handles as "missing".
    """
    node_values = np.zeros((len(graph.nodes), values.shape[1]))
    is_present = np.zeros(len(graph.nodes))
    pix = graph.nodes.get_indexer(present[ID_COL])
    node_values[pix] = values
    is_present[pix] = 1

    # Keep only the edges that lead to a present neighbour, one matrix row per transaction in `rows`
    nb = graph.nb_adj[graph.nodes.get_indexer(rows[ID_COL])].multiply(is_present[None, :]).tocsr()
    nb.eliminate_zeros()
    count = np.asarray(nb.sum(axis=1)).ravel()

    mean = (nb @ node_values) / np.maximum(count, 1)[:, None]
    mean[count == 0] = np.nan

    maxi = np.full(mean.shape, np.nan)
    has = np.where(count > 0)[0]
    if len(has):  # row-wise max over each row's neighbours (a CSR row's entries are stored together)
        maxi[has] = np.maximum.reduceat(node_values[nb.indices], nb.indptr[:-1][has], axis=0)

    cols = [f"nb_mean_{n}" for n in names] + [f"nb_max_{n}" for n in names]
    return pd.DataFrame(np.hstack([mean, maxi]), columns=cols, index=rows.index)


def fit_local_pca(pool: pd.DataFrame):
    """Standardise the 93 local features of `pool` and fit a 10-component PCA with fixed signs (no labels used)."""
    # "covariance_eigh" is the solver scikit-learn >= 1.5 picks for data of this shape. 
    # Naming it makes an older version stop with an error instead of giving slightly different components.
    pca = make_pipeline(StandardScaler(), PCA(N_COMPONENTS, svd_solver="covariance_eigh")).fit(pool[LOCAL_COLS])
    # A component's sign is arbitrary (scikit-learn versions before 1.5 chose it differently).
    # Fix it, so that "largest value among the neighbours" means the same thing on every version.
    # The change is made in place, so the returned PCA uses the fixed signs.
    comps = pca[-1].components_
    biggest = np.abs(comps).argmax(axis=1)                               # each component's largest weight ...
    comps *= np.sign(comps[np.arange(N_COMPONENTS), biggest])[:, None]  # ... is made positive
    return pca


def add_graph_features(graph: Graph, train_rows: pd.DataFrame, pred_rows: pd.DataFrame, pca) -> list:
    """Return copies of train_rows and pred_rows with the 27 graph columns (GRAPH_COLS) appended, one rule for both."""
    present = pd.concat([train_rows, pred_rows])  # present = training rows + prediction rows
    pcs = np.vstack([pca.transform(train_rows[LOCAL_COLS]), pca.transform(pred_rows[LOCAL_COLS])])
    return [pd.concat([rows,
                       presence_features(graph, rows, present[ID_COL]),
                       neighbour_mean_max(graph, rows, present, pcs, PC_COLS)], axis=1)
            for rows in (train_rows, pred_rows)]