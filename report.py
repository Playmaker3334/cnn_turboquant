import csv
import os
import torch
from typing import List, Dict


def print_table(rows: List[Dict], log):
    headers = list(rows[0].keys())
    col_w   = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]
    sep     = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    def pr(vals):
        print("  ".join(str(v).rjust(w) for v, w in zip(vals, col_w)), flush=True)

    pr(headers)
    print(sep)

    prev_cnn = None
    for row in rows:
        if prev_cnn is not None and row["cnn"] != prev_cnn:
            print(sep)
        pr(list(row.values()))
        prev_cnn = row["cnn"]


def save_csv(rows: List[Dict], meta: dict, quant_report: dict, path: str, log):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# BENCHMARK METADATA"])
        for k, v in meta.items():
            writer.writerow([f"# {k}", v])
        writer.writerow([])

        writer.writerow(["# QUANTIZATION VERIFICATION"])
        for k, v in quant_report.items():
            writer.writerow([f"# {k}", v])
        writer.writerow([])

        writer.writerow(["# BENCHMARK RESULTS"])
        writer.writerow(list(rows[0].keys()))
        for row in rows:
            writer.writerow(list(row.values()))

    size = os.path.getsize(path)
    log(f"  CSV saved to : {path}")
    log(f"  File size    : {size} bytes  |  rows: {len(rows)}")
