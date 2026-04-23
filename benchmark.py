"""
CNN TurboQuant Benchmark — entry point
=======================================
Trains two CNNs on CIFAR-10, extracts features, compresses with
float32 / int8 / tq_2bit / tq_3bit / tq_4bit and evaluates across
distortion, retrieval, classification, and efficiency metrics.

Usage
-----
  Kaggle / Colab:
    !pip install -q scipy
    %run benchmark.py

  Local / terminal:
    pip install -r requirements.txt
    python benchmark.py

  Download CSV after Colab run:
    from google.colab import files
    files.download("benchmark_results.csv")
"""

import numpy as np
import torch

from config  import (DEVICE, SEED, CSV_PATH,
                     CNN_A_LABEL, CNN_B_LABEL,
                     FEATURE_DIM, BIT_WIDTHS)
from data    import get_loaders
from trainer import train_cnn_a, train_cnn_b, extract_features
from runner  import run_all
from report  import print_table, save_csv
from quantizer import verify_quantization

torch.manual_seed(SEED)
np.random.seed(SEED)

SEP  = "=" * 90
SEP2 = "-" * 90

def log(msg=""): print(msg, flush=True)
def section(t):  log(f"\n{SEP}\n  {t}\n{SEP}")


# ─────────────────────────────────────────────
# 1. HARDWARE & QUANTIZATION VERIFICATION
# ─────────────────────────────────────────────
section("HARDWARE & QUANTIZATION VERIFICATION")

log(f"  CUDA available     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    log(f"  GPU name           : {props.name}")
    log(f"  GPU memory         : {props.total_memory / 1024**3:.1f} GB")
    log(f"  CUDA capability    : {props.major}.{props.minor}")
    _native_int8 = props.major >= 7
    log(f"  int8 native matmul : {'YES (Volta+ architecture)' if _native_int8 else 'NO  (software fallback)'}")
else:
    log("  GPU name           : N/A  (running on CPU)")
    _native_int8 = False

log()
log("  Running quantization self-tests ...")
quant_report = verify_quantization()

log()
log(f"  int8  torch.qint8 confirmed : {quant_report['int8_dtype_confirmed']}")
log(f"  int8  bytes per value       : {quant_report['int8_bytes_per_val']}")
log(f"  int8  dequantize MAE        : {quant_report['int8_mae']:.6f}")
log()
for b in BIT_WIDTHS:
    ok    = quant_report[f"tq_{b}bit_roundtrip_ok"]
    nbytes= quant_report[f"tq_{b}bit_bytes_per_vec"]
    ratio = quant_report[f"tq_{b}bit_compression_ratio"]
    log(f"  tq_{b}bit  roundtrip ok        : {ok}  |  {nbytes} bytes/vec  |  {ratio}x vs float32")

log()
log("  VERDICT:")
log(f"    int8    : {'REAL torch.qint8 storage' if quant_report['int8_dtype_confirmed'] else 'FAILED'}")
for b in BIT_WIDTHS:
    ok = quant_report[f"tq_{b}bit_roundtrip_ok"]
    log(f"    tq_{b}bit : {'REAL uint8 bit-packed storage' if ok else 'FAILED'}")
log(f"    compute : float32 after dequantize — honest on all hardware")

# ─────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────
section("DATA LOADING — CIFAR-10")
train_loader, val_loader, val_ds = get_loaders()
log(f"  Train samples : {50_000:,}")
log(f"  Val samples   : {len(val_ds):,}")
log(f"  Device        : {DEVICE}")

# ─────────────────────────────────────────────
# 3. CNN-A TRAINING
# ─────────────────────────────────────────────
section(f"TRAINING — {CNN_A_LABEL}")
log("  Standard CrossEntropyLoss + AdamW + CosineAnnealingLR")
log()
best_acc_a = train_cnn_a(train_loader, val_loader, "/tmp/cnn_a_best.pt", log)

# ─────────────────────────────────────────────
# 4. CNN-B TRAINING
# ─────────────────────────────────────────────
section(f"TRAINING — {CNN_B_LABEL}")
log("  LabelSmoothing + Mixup + AdamW + CosineAnnealingLR")
log()
best_acc_b = train_cnn_b(train_loader, val_loader, "/tmp/cnn_b_best.pt", log)

# ─────────────────────────────────────────────
# 5. FEATURE EXTRACTION
# ─────────────────────────────────────────────
section("FEATURE EXTRACTION")

log(f"\n{SEP2}\n  {CNN_A_LABEL}\n{SEP2}")
X_a, y_a = extract_features("/tmp/cnn_a_best.pt", val_loader, log)

log(f"\n{SEP2}\n  {CNN_B_LABEL}\n{SEP2}")
X_b, y_b = extract_features("/tmp/cnn_b_best.pt", val_loader, log)

# ─────────────────────────────────────────────
# 6. BENCHMARK
# ─────────────────────────────────────────────
section("BENCHMARKING ALL CONFIGS")

rows_a = run_all(X_a, y_a, CNN_A_LABEL, log)
rows_b = run_all(X_b, y_b, CNN_B_LABEL, log)
all_rows = rows_a + rows_b

# ─────────────────────────────────────────────
# 7. RESULTS TABLE
# ─────────────────────────────────────────────
section("RESULTS")
print_table(all_rows, log)

log()
log(f"  Shannon optimality factor : sqrt(3*pi/2) = {np.sqrt(3*np.pi/2):.4f}x")
log(f"  CNN-A best val accuracy   : {best_acc_a:.2f}%")
log(f"  CNN-B best val accuracy   : {best_acc_b:.2f}%")

# ─────────────────────────────────────────────
# 8. CSV EXPORT
# ─────────────────────────────────────────────
section("CSV EXPORT")

meta = {
    "epochs":          30,
    "feature_dim":     FEATURE_DIM,
    "seed":            SEED,
    "device":          str(DEVICE),
    "cuda_available":  torch.cuda.is_available(),
    "gpu_name":        torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU",
    "int8_verified":   quant_report["int8_dtype_confirmed"],
    "cnn_a_best_acc":  f"{best_acc_a:.2f}%",
    "cnn_b_best_acc":  f"{best_acc_b:.2f}%",
    "torch_version":   torch.__version__,
    "cnn_a_training":  "CrossEntropyLoss + AdamW + CosineAnnealingLR",
    "cnn_b_training":  "LabelSmoothing(0.1) + Mixup(0.4) + AdamW + CosineAnnealingLR",
}

save_csv(all_rows, meta, quant_report, CSV_PATH, log)
log()
log("  Done.")
