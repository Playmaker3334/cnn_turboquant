import numpy as np


def _score_matrix(X: np.ndarray, Xr: np.ndarray):
    scores = Xr @ Xr.T
    np.fill_diagonal(scores, -np.inf)
    true_scores = X @ X.T
    np.fill_diagonal(true_scores, -np.inf)
    return scores, true_scores


def recall_at_k(X: np.ndarray, Xr: np.ndarray, k: int) -> float:
    """Fraction of queries where the exact nearest neighbor appears in top-k."""
    scores, true_scores = _score_matrix(X, Xr)
    true_nn = np.argmax(true_scores, axis=1)
    top_k   = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean([true_nn[i] in top_k[i] for i in range(len(true_nn))]))


def mean_reciprocal_rank(X: np.ndarray, Xr: np.ndarray) -> float:
    """MRR: E[1 / rank(true_nn)] in the approximated ranking."""
    scores, true_scores = _score_matrix(X, Xr)
    true_nn = np.argmax(true_scores, axis=1)
    order   = np.argsort(-scores, axis=1)
    rr = []
    for i in range(len(true_nn)):
        rank = np.where(order[i] == true_nn[i])[0]
        rr.append(1.0 / (rank[0] + 1) if len(rank) > 0 else 0.0)
    return float(np.mean(rr))


def map_at_k(X: np.ndarray, Xr: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Mean Average Precision at k using class labels as relevance signal."""
    scores = Xr @ Xr.T
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=1)[:, :k]
    aps = []
    for i in range(len(y)):
        hits, prec_sum = 0, 0.0
        for j, idx in enumerate(order[i]):
            if y[idx] == y[i]:
                hits     += 1
                prec_sum += hits / (j + 1)
        aps.append(prec_sum / min(hits, k) if hits > 0 else 0.0)
    return float(np.mean(aps))
