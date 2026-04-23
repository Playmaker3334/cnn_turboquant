import time
import numpy as np
from config import N_BENCH


def measure_throughput(fn, X: np.ndarray, n: int = N_BENCH) -> float:
    """Vectors processed per second (median of n runs)."""
    fn(X[:10])  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(X)
        times.append(time.perf_counter() - t0)
    return len(X) / float(np.median(times))


def measure_query_latency_ms(fn, n: int = N_BENCH) -> float:
    """
    Median latency in milliseconds for a single query operation.
    fn must accept no arguments and perform one full query cycle.
    """
    fn()  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000
