import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from backbone import CNN
from config   import (EPOCHS, LR, WEIGHT_DECAY, DEVICE, SEED,
                      LABEL_SMOOTH, MIXUP_ALPHA)


# ─────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────

def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    """
    Apply mixup augmentation to a batch.
    Returns mixed images, label_a, label_b, lambda.
    """
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────

def _train_epoch_standard(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


def _train_epoch_mixup(model, loader, criterion, optimizer, alpha: float):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        mixed, la, lb, lam = mixup_batch(imgs, labels, alpha)
        optimizer.zero_grad()
        out  = model(mixed)
        loss = lam * criterion(out, la) + (1 - lam) * criterion(out, lb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == la).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


def _validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total * 100


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def train_cnn_a(train_loader, val_loader, ckpt_path: str, log) -> float:
    """
    CNN-A: standard CrossEntropyLoss + AdamW + CosineAnnealingLR.
    Returns best val accuracy.
    """
    torch.manual_seed(SEED)
    model     = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    log(f"  Params          : {model.n_params():,}")
    log(f"  Loss            : CrossEntropyLoss (no smoothing)")
    log(f"  Augmentation    : RandomCrop + HorizontalFlip")
    log(f"  Optimizer       : AdamW  lr={LR}  wd={WEIGHT_DECAY}")
    log()
    _print_epoch_header(log)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_l, tr_a   = _train_epoch_standard(model, train_loader, criterion, optimizer)
        val_l, val_a = _validate(model, val_loader, criterion)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0
        marker = " *" if val_a > best_acc else ""
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), ckpt_path)
        log(f"  {epoch:>5}  {tr_l:>10.4f}  {tr_a:>8.2f}%  {val_l:>8.4f}  {val_a:>6.2f}%  {lr:>10.2e}  {elapsed:>5.1f}s{marker}")

    log()
    log(f"  Best val accuracy : {best_acc:.2f}%")
    return best_acc


def train_cnn_b(train_loader, val_loader, ckpt_path: str, log) -> float:
    """
    CNN-B: LabelSmoothing + Mixup + AdamW + CosineAnnealingLR.
    Produces more uniformly distributed embeddings, often better for retrieval.
    Returns best val accuracy.
    """
    torch.manual_seed(SEED)
    model     = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    log(f"  Params          : {model.n_params():,}")
    log(f"  Loss            : CrossEntropyLoss (label_smoothing={LABEL_SMOOTH})")
    log(f"  Augmentation    : RandomCrop + HorizontalFlip + ColorJitter + Mixup(alpha={MIXUP_ALPHA})")
    log(f"  Optimizer       : AdamW  lr={LR}  wd={WEIGHT_DECAY}")
    log()
    _print_epoch_header(log)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_l, tr_a   = _train_epoch_mixup(model, train_loader, criterion, optimizer, MIXUP_ALPHA)
        val_l, val_a = _validate(model, val_loader, criterion)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0
        marker = " *" if val_a > best_acc else ""
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), ckpt_path)
        log(f"  {epoch:>5}  {tr_l:>10.4f}  {tr_a:>8.2f}%  {val_l:>8.4f}  {val_a:>6.2f}%  {lr:>10.2e}  {elapsed:>5.1f}s{marker}")

    log()
    log(f"  Best val accuracy : {best_acc:.2f}%")
    return best_acc


def extract_features(ckpt_path: str, val_loader, log) -> tuple:
    """
    Load best checkpoint and extract L2-normalized float32 features
    from the full val set. Returns (X_f32, y_labels).
    """
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            f    = model(imgs, return_features=True)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())

    X   = np.concatenate(feats,  axis=0).astype(np.float32)
    y   = np.concatenate(labels, axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X   = X / np.maximum(norms, 1e-12)

    log(f"  Feature shape   : {X.shape}  dtype={X.dtype}")
    log(f"  float32 size    : {X.nbytes / 1024**2:.3f} MB")
    return X, y


def _print_epoch_header(log):
    log(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  {'Time':>6}")
    log(f"  {'-' * 76}")
