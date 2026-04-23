import numpy as np


def relative_mse(X: np.ndarray, Xr: np.ndarray) -> float:
    """Mean ||x - x_hat||^2 / ||x||^2 over all vectors."""
    num   = np.sum((X - Xr) ** 2, axis=1)
    denom = np.sum(X ** 2, axis=1) + 1e-14
    return float(np.mean(num / denom))


def cosine_preservation(X: np.ndarray, Xr: np.ndarray) -> float:
    """Mean cosine similarity between original and reconstructed vectors."""
    cos = np.sum(X * Xr, axis=1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(Xr, axis=1) + 1e-14
    )
    return float(np.mean(cos))


def l2_error(X: np.ndarray, Xr: np.ndarray) -> float:
    """Mean L2 distance between original and reconstructed vectors."""
    return float(np.mean(np.linalg.norm(X - Xr, axis=1)))


def ip_stats(X: np.ndarray, Xr: np.ndarray, Y: np.ndarray):
    """
    Inner product estimation quality: <x_hat, y> vs <x, y>.
    Returns (MAE, bias, std).
    """
    err = np.einsum('ij,ij->i', Xr, Y) - np.einsum('ij,ij->i', X, Y)
    return float(np.mean(np.abs(err))), float(np.mean(err)), float(np.std(err))


def shannon_bound(n_bits: int) -> float:
    """Shannon rate-distortion lower bound for unit Gaussian vectors: 4^(-b)."""
    return 4.0 ** (-n_bits)
