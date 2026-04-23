import os
import tempfile
import torch

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────
EPOCHS       = 30
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
FEATURE_DIM  = 256
NUM_CLASSES  = 10
SEED         = 42

# ─────────────────────────────────────────────
# Benchmark settings
# ─────────────────────────────────────────────
BIT_WIDTHS = [2, 3, 4]
K_VALUES   = [1, 5, 10]
N_BENCH    = 5

# ─────────────────────────────────────────────
# Paths
#   Checkpoints go to the OS temp dir so the code runs on
#   Linux (/tmp), macOS (/var/folders/...) and Windows (%TEMP%)
#   without modification.
# ─────────────────────────────────────────────
DATA_DIR = "./data"
CSV_PATH = "./benchmark_results.csv"

_CKPT_DIR         = tempfile.gettempdir()
CHECKPOINT_A_PATH = os.path.join(_CKPT_DIR, "cnn_a_best.pt")
CHECKPOINT_B_PATH = os.path.join(_CKPT_DIR, "cnn_b_best.pt")

# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# CNN A — standard training
# ─────────────────────────────────────────────
CNN_A_NAME  = "cnn_standard"
CNN_A_LABEL = "CNN-A (standard)"

# ─────────────────────────────────────────────
# CNN B — label smoothing + mixup
# ─────────────────────────────────────────────
CNN_B_NAME   = "cnn_regularized"
CNN_B_LABEL  = "CNN-B (label_smooth+mixup)"
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.4