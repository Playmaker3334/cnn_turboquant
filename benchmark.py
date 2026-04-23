import numpy as np
import torch

from config  import (DEVICE, SEED, CSV_PATH,
                     CNN_A_LABEL, CNN_B_LABEL,
                     CHECKPOINT_A_PATH, CHECKPOINT_B_PATH,
                     FEATURE_DIM, BIT_WIDTHS)
from data    import get_loaders
from trainer import train_cnn_a, train_cnn_b, extract_features
from runner  import run_all
from report  import print_table, save_csv
from quantizer import verify_quantization

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# CUDA performance flags
#   TF32 keeps FP32 range with ~10-bit mantissa on Ampere+ GPUs:
#   ~1.5x faster matmul, negligible accuracy impact for CNN training.
#   cudnn.benchmark auto-selects fastest conv kernels after warmup.
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

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
    log(f"  TF32 matmul        : ENABLED")
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
log("  TurboQuant_mse  (Algorithm 1, MSE-optimal):")
for b in BIT_WIDTHS:
    ok     = quant_report[f"tq_{b}bit_roundtrip_ok"]
    nbytes = quant_report[f"tq_{b}bit_bytes_per_vec"]
    ratio  = quant_report[f"tq_{b}bit_compression_ratio"]
    log(f"    tq_{b}bit        : roundtrip={ok}  |  {nbytes} bytes/vec  |  {ratio}x vs float32")
log()
log("  TurboQuant_prod (Theorem 2, unbiased inner-product):")
for b in BIT_WIDTHS:
    if b < 2:
        continue
    nbytes  = quant_report[f"tq_prod_{b}bit_bytes_per_vec"]
    ratio   = quant_report[f"tq_prod_{b}bit_compression_ratio"]
    ip_bias = quant_report[f"tq_prod_{b}bit_ip_bias"]
    ip_mae  = quant_report[f"tq_prod_{b}bit_ip_mae"]
    log(f"    tq_{b}bit_prod   : {nbytes} bytes/vec  |  {ratio}x  |  ip_bias={ip_bias:+.6f}  ip_mae={ip_mae:.6f}")

log()
log("  VERDICT:")
log(f"    int8         : {'REAL torch.qint8 storage' if quant_report['int8_dtype_confirmed'] else 'FAILED'}")
for b in BIT_WIDTHS:
    ok = quant_report[f"tq_{b}bit_roundtrip_ok"]
    log(f"    tq_{b}bit      : {'REAL uint8 bit-packed storage' if ok else 'FAILED'}")
for b in BIT_WIDTHS:
    if b < 2:
        continue
    ok = quant_report[f"tq_prod_{b}bit_roundtrip_ok"]
    log(f"    tq_{b}bit_prod : {'REAL mse+QJL bit-packed storage' if ok else 'FAILED'}")
log(f"    compute      : float32 after dequantize - honest on all hardware")

# ─────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────
section("DATA LOADING - CIFAR-10")
train_loader, val_loader, val_ds = get_loaders()
log(f"  Train samples : {50_000:,}")
log(f"  Val samples   : {len(val_ds):,}")
log(f"  Device        : {DEVICE}")

# ─────────────────────────────────────────────
# 3. CNN-A TRAINING
# ─────────────────────────────────────────────
section(f"TRAINING - {CNN_A_LABEL}")
log("  Standard CrossEntropyLoss + AdamW + CosineAnnealingLR")
log()
best_acc_a = train_cnn_a(train_loader, val_loader, CHECKPOINT_A_PATH, log)

# ─────────────────────────────────────────────
# 4. CNN-B TRAINING
# ─────────────────────────────────────────────
section(f"TRAINING - {CNN_B_LABEL}")
log("  LabelSmoothing + Mixup + AdamW + CosineAnnealingLR")
log()
best_acc_b = train_cnn_b(train_loader, val_loader, CHECKPOINT_B_PATH, log)

# ─────────────────────────────────────────────
# 5. FEATURE EXTRACTION
# ─────────────────────────────────────────────
section("FEATURE EXTRACTION")

log(f"\n{SEP2}\n  {CNN_A_LABEL}\n{SEP2}")
X_a, y_a = extract_features(CHECKPOINT_A_PATH, val_loader, log)

log(f"\n{SEP2}\n  {CNN_B_LABEL}\n{SEP2}")
X_b, y_b = extract_features(CHECKPOINT_B_PATH, val_loader, log)

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
log(f"  Distortion gap to Shannon lower bound : ~2.7x (Zandieh et al. 2025, Theorem 1)")
log(f"  CNN-A best val accuracy               : {best_acc_a:.2f}%")
log(f"  CNN-B best val accuracy               : {best_acc_b:.2f}%")

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
    "tf32_enabled":    torch.cuda.is_available(),
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