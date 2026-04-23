import numpy as np
import time
from typing import List, Dict

from quantizer import (
    turboquant_compress, turboquant_decompress,
    turboquant_prod_compress, turboquant_prod_decompress,
    int8_compress, int8_decompress, CODEBOOKS,
)
from metrics.distortion     import relative_mse, cosine_preservation, l2_error, ip_stats, shannon_bound
from metrics.retrieval      import recall_at_k, mean_reciprocal_rank, map_at_k
from metrics.classification import knn_accuracy
from metrics.efficiency     import measure_throughput, measure_query_latency_ms
from config import SEED, FEATURE_DIM


# Config schema: (display_name, mode, sbits, variant)
#   mode    : None   -> float32 baseline
#             "int8" -> torch.qint8
#             int n  -> TurboQuant at n effective bits per coord
#   variant : None | "mse" | "prod"
#   sbits   : nominal effective bits per coord  (used for Shannon bound and ratio)
CONFIGS = [
    ("float32",      None,   32, None),
    ("int8",         "int8",  8, None),
    ("tq_2bit",      2,       2, "mse"),
    ("tq_3bit",      3,       3, "mse"),
    ("tq_4bit",      4,       4, "mse"),
    ("tq_2bit_prod", 2,       2, "prod"),
    ("tq_3bit_prod", 3,       3, "prod"),
    ("tq_4bit_prod", 4,       4, "prod"),
]


def run_all(X_f32: np.ndarray, y: np.ndarray, cnn_label: str, log) -> List[Dict]:
    """
    Runs all quantization configs on X_f32 and returns a list of result dicts.
    X_f32 must be L2-normalized float32, shape (N, D).
    """
    SEP2 = "-" * 90

    # Y vectors for inner product evaluation
    Y_raw = np.random.RandomState(SEED + 1).randn(*X_f32.shape).astype(np.float32)
    Y     = Y_raw / np.maximum(np.linalg.norm(Y_raw, axis=1, keepdims=True), 1e-12)

    # subsample for retrieval (2000 keeps O(N^2) matrix tractable on CPU)
    idx = np.random.RandomState(SEED).choice(len(X_f32), min(2000, len(X_f32)), replace=False)
    Xs  = X_f32[idx];  ys = y[idx]

    rows = []

    for name, mode, sbits, variant in CONFIGS:
        log(f"\n{SEP2}")
        log(f"  {cnn_label}  |  {name}")
        log(SEP2)

        # ── [1] compress
        log("  [1/6] Compressing ...")
        t0 = time.perf_counter()

        if mode is None:
            X_stored    = X_f32.copy()
            mem_bytes   = X_stored.nbytes
            compress_fn = lambda X: X.copy()
            decompress  = lambda: X_stored

        elif mode == "int8":
            qt          = int8_compress(X_f32)
            mem_bytes   = X_f32.shape[0] * X_f32.shape[1] * 1
            compress_fn = lambda X: int8_compress(X)
            decompress  = lambda: int8_decompress(qt)
            log(f"       dtype={qt.dtype}  scale={qt.q_scale():.6f}  zp={qt.q_zero_point()}")

        elif variant == "mse":
            packed, norms_stored = turboquant_compress(X_f32, mode)
            mem_bytes   = packed.nbytes + norms_stored.nbytes
            compress_fn = lambda X, b=mode: turboquant_compress(X, b)
            decompress  = (
                lambda p=packed, ns=norms_stored, b=mode:
                turboquant_decompress(p, ns, b)
            )
            log(f"       packed={packed.shape} dtype={packed.dtype}  norms={norms_stored.shape}")

        elif variant == "prod":
            mse_pk, sign_pk, norms_stored = turboquant_prod_compress(X_f32, mode)
            mem_bytes   = mse_pk.nbytes + sign_pk.nbytes + norms_stored.nbytes
            compress_fn = lambda X, b=mode: turboquant_prod_compress(X, b)
            decompress  = (
                lambda mp=mse_pk, sp=sign_pk, ns=norms_stored, b=mode:
                turboquant_prod_decompress(mp, sp, ns, b)
            )
            log(
                f"       mse_packed={mse_pk.shape}  "
                f"sign_packed={sign_pk.shape}  norms={norms_stored.shape}"
            )

        else:
            raise ValueError(f"Unknown config: {name}")

        log(f"       time={time.perf_counter()-t0:.3f}s  real_mem={mem_bytes/1024**2:.4f} MB")

        # ── [2] decompress
        log("  [2/6] Decompressing ...")
        t0    = time.perf_counter()
        X_rec = decompress()
        log(f"       shape={X_rec.shape}  dtype={X_rec.dtype}  time={time.perf_counter()-t0:.3f}s")

        # ── [3] distortion
        log("  [3/6] Distortion metrics ...")
        rel_mse_v = relative_mse(X_f32, X_rec)
        cos_pres  = cosine_preservation(X_f32, X_rec)
        l2_err    = l2_error(X_f32, X_rec)
        ip_mae, ip_bias, ip_std = ip_stats(X_f32, X_rec, Y)

        # Shannon bound only meaningful for TurboQuant modes
        if isinstance(mode, int):
            bound = shannon_bound(sbits)
            ratio = rel_mse_v / bound
        else:
            bound = float("nan")
            ratio = float("nan")

        log(f"       rel_mse={rel_mse_v:.6f}  cos_pres={cos_pres:.6f}  l2={l2_err:.6f}")
        log(f"       ip_mae={ip_mae:.6f}  ip_bias={ip_bias:.6f}  ip_std={ip_std:.6f}")
        if isinstance(mode, int):
            log(f"       shannon_bound={bound:.6f}  mse/bound={ratio:.4f}x")

        # ── [4] retrieval
        log("  [4/6] Retrieval metrics (N=2000 subsample) ...")
        Xrs   = X_rec[idx]
        r1    = recall_at_k(Xs, Xrs, 1)
        r5    = recall_at_k(Xs, Xrs, 5)
        r10   = recall_at_k(Xs, Xrs, 10)
        mrr_v = mean_reciprocal_rank(Xs, Xrs)
        map10 = map_at_k(Xs, Xrs, ys, k=10)
        knn1  = knn_accuracy(Xs, Xrs, ys, 1)
        knn5  = knn_accuracy(Xs, Xrs, ys, 5)
        log(f"       recall@1={r1*100:.1f}%  @5={r5*100:.1f}%  @10={r10*100:.1f}%")
        log(f"       mrr={mrr_v:.4f}  map@10={map10:.4f}")
        log(f"       knn@1={knn1*100:.2f}%  knn@5={knn5*100:.2f}%")

        # ── [5] throughput
        log("  [5/6] Throughput ...")
        tput = measure_throughput(compress_fn, X_f32)
        log(f"       {tput:,.0f} vecs/sec")

        # ── [6] query latency
        log("  [6/6] Query latency ...")
        q = X_f32[:1]
        if mode is None:
            qfn = lambda: X_stored @ q.T
        elif mode == "int8":
            qfn = lambda: int8_decompress(qt) @ q.T
        elif variant == "mse":
            qfn = (
                lambda p=packed, ns=norms_stored, b=mode:
                turboquant_decompress(p, ns, b) @ q.T
            )
        elif variant == "prod":
            qfn = (
                lambda mp=mse_pk, sp=sign_pk, ns=norms_stored, b=mode:
                turboquant_prod_decompress(mp, sp, ns, b) @ q.T
            )
        lat_ms = measure_query_latency_ms(qfn)
        log(f"       {lat_ms:.3f} ms/query")

        rows.append({
            "cnn":         cnn_label,
            "method":      name,
            "mem_mb":      f"{mem_bytes/1024**2:.4f}",
            "compression": f"{32/sbits:.1f}x",
            "rel_mse":     f"{rel_mse_v:.6f}",
            "mse/bound":   f"{ratio:.4f}" if isinstance(mode, int) else "---",
            "cos_pres":    f"{cos_pres:.6f}",
            "l2_err":      f"{l2_err:.6f}",
            "ip_mae":      f"{ip_mae:.6f}",
            "ip_bias":     f"{ip_bias:.6f}",
            "ip_std":      f"{ip_std:.6f}",
            "r@1":         f"{r1*100:.1f}%",
            "r@5":         f"{r5*100:.1f}%",
            "r@10":        f"{r10*100:.1f}%",
            "mrr":         f"{mrr_v:.4f}",
            "map@10":      f"{map10:.4f}",
            "knn@1":       f"{knn1*100:.2f}%",
            "knn@5":       f"{knn5*100:.2f}%",
            "vecs/sec":    f"{tput:,.0f}",
            "lat_ms":      f"{lat_ms:.3f}",
        })

    return rows