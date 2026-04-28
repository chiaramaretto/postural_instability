import os
import math
import torch
import numpy as np
from typing import Optional
from torch import nn


def _collect_predictions(model, loader, device):
    model.eval()
    ys = []
    preds = []
    groups = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            ys.append(y.detach().cpu().numpy())
            preds.append(out.detach().cpu().numpy())
            groups.extend(batch.get("group", []))
    if ys:
        y = np.concatenate(ys, axis=0)
        p = np.concatenate(preds, axis=0)
    else:
        y = np.array([])
        p = np.array([])
    return y, p, groups


def _aggregate_by_group(y, p, groups):
    import pandas as pd
    df = pd.DataFrame({"y": y, "p": p, "group": groups})
    agg = df.groupby("group").mean()
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def evaluate_model(model, loader, device, aggregate_by_group=False):
    y, p_logits, groups = _collect_predictions(model, loader, device)
    if len(y) == 0:
        return {"acc": float('nan'), "n": 0}

    # predictions: logits -> class
    p = np.argmax(p_logits, axis=1)
    if aggregate_by_group and len(groups) > 0:
        y, p = _aggregate_by_group(y, p, groups)

    acc = float((y == p).mean())
    return {"acc": acc, "n": len(np.unique(groups)) if aggregate_by_group else len(y)}


def train_epoch(model, loader, optimizer, loss_fn, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device).long()
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += float(loss.detach().cpu().numpy()) * x.shape[0]
        n += x.shape[0]
    return total_loss / max(1, n)


def fit_model(model: nn.Module, train_loader, val_loader, device, max_epochs=50, patience=8, min_delta=1e-3, scheduler_patience=3, scheduler_factor=0.5, min_lr=1e-6, best_path: Optional[str] = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)

    best_metric = -math.inf
    epochs_no_improve = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate_model(model, val_loader, device, aggregate_by_group=True)
        val_acc = metrics["acc"]
        scheduler.step(val_acc)

        improved = (val_acc - best_metric) > min_delta
        if improved:
            best_metric = val_acc
            epochs_no_improve = 0
            if best_path:
                torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            break

    # load best
    if best_path and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model
