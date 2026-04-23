import torch

EPOCHS      = 30
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY = 1e-4
FEATURE_DIM = 256
NUM_CLASSES = 10
SEED        = 42
BIT_WIDTHS  = [2, 3, 4]
K_VALUES    = [1, 5, 10]
N_BENCH     = 5
DATA_DIR    = "./data"
CSV_PATH    = "./benchmark_results.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN A — standard training
CNN_A_NAME  = "cnn_standard"
CNN_A_LABEL = "CNN-A (standard)"

# CNN B — label smoothing + mixup
CNN_B_NAME   = "cnn_regularized"
CNN_B_LABEL  = "CNN-B (label_smooth+mixup)"
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.4
