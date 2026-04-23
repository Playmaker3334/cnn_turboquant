# CNN TurboQuant Benchmark

Benchmarks TurboQuant vector quantization (arXiv:2504.19874) applied to CNN
feature embeddings extracted from CIFAR-10. Trains two CNNs with different
training regimes and evaluates how each quantization method affects retrieval
and classification quality.

## Two CNNs compared

| Model   | Training regime                                          |
|---------|----------------------------------------------------------|
| CNN-A   | Standard CrossEntropyLoss + AdamW + CosineAnnealingLR   |
| CNN-B   | LabelSmoothing(0.1) + Mixup(alpha=0.4) + AdamW + CosineAnnealingLR |

Both share the same architecture (3x ConvBlock, 256-dim feature layer) and
are trained on CIFAR-10 for 30 epochs. Features are extracted from the val
set and compressed with each method.

## Quantization methods

| Method   | Storage              | Real?                        |
|----------|----------------------|------------------------------|
| float32  | float32 numpy        | baseline                     |
| int8     | torch.qint8          | real 1-byte storage          |
| tq_2bit  | uint8 bit-packed     | real — 64 bytes per 256-dim vector  |
| tq_3bit  | uint8 bit-packed     | real — 96 bytes per 256-dim vector  |
| tq_4bit  | uint8 bit-packed     | real — 128 bytes per 256-dim vector |

## Metrics evaluated

| Block          | Metrics                                                  |
|----------------|----------------------------------------------------------|
| Distortion     | rel_mse, mse/shannon_bound, cosine_preservation, L2 error |
| Inner product  | ip_mae, ip_bias, ip_std                                  |
| Retrieval      | Recall@1/5/10, MRR, mAP@10                              |
| Classification | KNN@1 accuracy, KNN@5 accuracy                           |
| Efficiency     | mem_mb (real nbytes), compression ratio, vecs/sec, latency_ms |

## Project structure

```
cnn_turboquant_benchmark/
├── benchmark.py       entry point
├── config.py          all hyperparameters
├── data.py            CIFAR-10 dataloaders
├── backbone.py        CNN architecture (shared)
├── trainer.py         train_cnn_a, train_cnn_b, extract_features
├── quantizer.py       TurboQuant + bit-packing + int8 + verification
├── runner.py          runs all configs for one CNN's features
├── report.py          print_table + save_csv
├── metrics/
│   ├── __init__.py
│   ├── distortion.py  rel_mse, cosine_preservation, l2_error, ip_stats
│   ├── retrieval.py   recall_at_k, mrr, map_at_k
│   ├── classification.py  knn_accuracy
│   └── efficiency.py  throughput, query_latency
├── requirements.txt
└── README.md
```

## Usage

### Kaggle / Google Colab

```bash
git clone https://github.com/your-user/cnn_turboquant_benchmark.git
cd cnn_turboquant_benchmark
pip install scipy
python benchmark.py
```

Download CSV after run (Colab):
```python
from google.colab import files
files.download("benchmark_results.csv")
```

### Local

```bash
git clone https://github.com/your-user/cnn_turboquant_benchmark.git
cd cnn_turboquant_benchmark
pip install -r requirements.txt
python benchmark.py
```

## Hardware compatibility

| GPU       | int8 real | bit-pack | Notes                        |
|-----------|-----------|----------|------------------------------|
| T4        | yes       | yes      | Kaggle / Colab free          |
| P100      | yes       | yes      | Kaggle                       |
| A100      | yes       | yes      | Colab Pro+                   |
| RTX 4060  | yes       | yes      | local                        |
| CPU only  | yes       | yes      | slower training, same output |

A VERDICT block prints at startup confirming storage type per method.

## Expected runtime

| Hardware  | Approx. time  |
|-----------|---------------|
| T4        | ~35 min       |
| A100      | ~18 min       |
| RTX 4060  | ~25 min       |
| CPU       | ~3-4 hours    |

## Reference

Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. (2025).
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
arXiv:2504.19874
