import numpy as np
import scipy.stats as stats
import torch
from typing import Tuple, Dict
from config import FEATURE_DIM, BIT_WIDTHS, SEED


# ── rotation matrix (fixed, shared across all calls)
_A = np.random.RandomState(SEED).randn(FEATURE_DIM, FEATURE_DIM).astype(np.float64)
R, _ = np.linalg.qr(_A)
R = R.astype(np.float32)


def _lloyd_max(n_bits: int, n_iter: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max optimal scalar quantizer for N(0,1)."""
    n  = 2 ** n_bits
    c  = stats.norm.ppf((np.arange(n) + 0.5) / n).astype(np.float64)
    for _ in range(n_iter):
        b  = (c[:-1] + c[1:]) / 2.0
        ab = np.concatenate([[-np.inf], b, [np.inf]])
        nc = np.empty(n)
        for i in range(n):
            lo, hi = ab[i], ab[i + 1]
            p = stats.norm.cdf(hi) - stats.norm.cdf(lo)
            nc[i] = (stats.norm.pdf(lo) - stats.norm.pdf(hi)) / p if p > 1e-14 else (lo + hi) / 2.0
        if np.max(np.abs(nc - c)) < 1e-12:
            break
        c = nc
    b = (c[:-1] + c[1:]) / 2.0
    return b.astype(np.float32), c.astype(np.float32)


CODEBOOKS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    b: _lloyd_max(b) for b in BIT_WIDTHS
}


# ─────────────────────────────────────────────────────────────
# BIT PACKING (real uint8 storage)
# ─────────────────────────────────────────────────────────────

def pack_bits(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Pack integer indices (shape N x D, values in [0, 2^n_bits))
    into a uint8 array using bit-shifting.
    Real memory footprint: N * ceil(D * n_bits / 8) bytes.
    """
    N, D       = indices.shape
    n_bytes    = (D * n_bits + 7) // 8
    packed     = np.zeros((N, n_bytes), dtype=np.uint8)
    for d in range(D):
        bit_pos  = d * n_bits
        byte_pos = bit_pos // 8
        bit_off  = bit_pos %  8
        vals     = indices[:, d].astype(np.uint32)
        rem, sh, bp = n_bits, bit_off, byte_pos
        while rem > 0:
            bh = min(8 - sh, rem)
            packed[:, bp] |= ((vals & ((1 << bh) - 1)) << sh).astype(np.uint8)
            vals >>= bh; rem -= bh; sh = 0; bp += 1
    return packed


def unpack_bits(packed: np.ndarray, D: int, n_bits: int) -> np.ndarray:
    """Reverse of pack_bits. Returns uint32 indices of shape N x D."""
    N       = packed.shape[0]
    indices = np.zeros((N, D), dtype=np.uint32)
    mask    = (1 << n_bits) - 1
    for d in range(D):
        bit_pos  = d * n_bits
        byte_pos = bit_pos // 8
        bit_off  = bit_pos %  8
        rem, sh, bp, os_ = n_bits, bit_off, byte_pos, 0
        while rem > 0:
            bh = min(8 - sh, rem)
            indices[:, d] |= (((packed[:, bp].astype(np.uint32) >> sh) & ((1 << bh) - 1)) << os_)
            rem -= bh; os_ += bh; sh = 0; bp += 1
    return indices & mask


# ─────────────────────────────────────────────────────────────
# TURBOQUANT
# ─────────────────────────────────────────────────────────────

def turboquant_compress(X: np.ndarray, n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full TurboQuant pipeline:
      1. L2-normalize
      2. Random orthogonal rotation
      3. Scale to ~N(0,1)
      4. Lloyd-Max scalar quantization per coordinate
      5. Bit-pack into uint8

    Returns:
      packed : uint8 array of shape (N, ceil(D*n_bits/8))  — real compressed storage
      norms  : float32 array of shape (N, 1)               — original norms
    """
    bounds, _ = CODEBOOKS[n_bits]
    norms     = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    X_rot     = (X / norms) @ R.T * np.sqrt(FEATURE_DIM)
    indices   = np.digitize(X_rot, bounds).astype(np.uint32)
    packed    = pack_bits(indices, n_bits)
    return packed, norms.astype(np.float32)


def turboquant_decompress(packed: np.ndarray, norms: np.ndarray, n_bits: int) -> np.ndarray:
    """Unpack + dequantize + unrotate -> float32 features."""
    _, cents = CODEBOOKS[n_bits]
    indices  = unpack_bits(packed, FEATURE_DIM, n_bits)
    X_q      = cents[indices] / np.sqrt(FEATURE_DIM)
    return (X_q @ R) * norms


# ─────────────────────────────────────────────────────────────
# INT8 REAL QUANTIZATION (torch.qint8)
# ─────────────────────────────────────────────────────────────

def int8_compress(X: np.ndarray):
    """
    Real int8 quantization via torch.quantize_per_tensor.
    Stored as torch.qint8 (1 byte per element).
    Returns quantized tensor.
    """
    t     = torch.from_numpy(X)
    scale = float(np.abs(X).max()) / 127.0
    return torch.quantize_per_tensor(t, scale=scale, zero_point=0, dtype=torch.qint8)


def int8_decompress(qt) -> np.ndarray:
    """Dequantize torch.qint8 tensor back to float32 numpy array."""
    return qt.dequantize().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────

def verify_quantization() -> dict:
    """
    Runs self-tests on bit packing and int8 to confirm
    real storage is working. Returns a dict of results.
    """
    report = {}

    # int8
    X_test   = np.random.RandomState(0).randn(32, FEATURE_DIM).astype(np.float32)
    qt       = int8_compress(X_test)
    Xr       = int8_decompress(qt)
    report["int8_dtype_confirmed"] = (qt.dtype == torch.qint8)
    report["int8_bytes_per_val"]   = 1
    report["int8_mae"]             = float(np.abs(X_test - Xr).mean())
    report["int8_real_storage"]    = report["int8_dtype_confirmed"]

    # bit packing per bit-width
    for b in BIT_WIDTHS:
        idx   = np.random.randint(0, 2**b, (64, FEATURE_DIM), dtype=np.uint32)
        pk    = pack_bits(idx, b)
        back  = unpack_bits(pk, FEATURE_DIM, b)
        ok    = bool(np.all(idx == back))
        n_bytes_per_vec = (FEATURE_DIM * b + 7) // 8
        report[f"tq_{b}bit_roundtrip_ok"]    = ok
        report[f"tq_{b}bit_bytes_per_vec"]   = n_bytes_per_vec
        report[f"tq_{b}bit_compression_ratio"] = round((FEATURE_DIM * 4) / n_bytes_per_vec, 2)

    return report
