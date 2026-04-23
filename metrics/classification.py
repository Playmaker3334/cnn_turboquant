import numpy as np
from config import NUM_CLASSES


def knn_accuracy(X: np.ndarray, Xr: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    KNN classification accuracy using compressed features Xr for retrieval.
    Ground truth labels come from y. Query set = index set (leave-self-out via -inf diagonal).
    """
    scores = Xr @ Xr.T
    np.fill_diagonal(scores, -np.inf)
    top_k  = np.argpartition(-scores, k, axis=1)[:, :k]
    preds  = np.array([
        np.bincount(y[top_k[i]], minlength=NUM_CLASSES).argmax()
        for i in range(len(y))
    ])
    return float(np.mean(preds == y))
