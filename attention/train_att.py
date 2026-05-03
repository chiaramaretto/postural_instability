import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, tid, mask, y = batch["x"].to(device), batch["task_ids"].to(device), batch["window_mask"].to(device), batch["y"].to(device)
        optimizer.zero_grad()
        pred, _ = model(x, tid, mask)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    for batch in loader:
        pred, _ = model(batch["x"].to(device), batch["task_ids"].to(device), batch["window_mask"].to(device))
        all_preds.extend(pred.cpu().numpy())
        all_y.extend(batch["y"].numpy())

    all_preds = np.array(all_preds, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    mae = float(np.mean(np.abs(all_preds - all_y)))
    rmse = float(np.sqrt(np.mean((all_preds - all_y) ** 2)))
    pred_clipped = np.clip(np.round(all_preds), 0, 4).astype(int)
    y_true = np.clip(np.round(all_y), 0, 4).astype(int)
    acc_rounded = float((pred_clipped == y_true).mean())
    
    return {
        "mae": mae,
        "rmse": rmse,
        "acc_rounded": acc_rounded,
    }

def fit_model(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=80,
    patience=12,
    min_delta=1e-3,
    scheduler_patience=5,
    scheduler_factor=0.5,
    min_lr=1e-5,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    criterion = nn.MSELoss()
    best_mae = float("inf")
    best_path = "best_regression_model.pth"
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["mae"])

        if metrics["mae"] < (best_mae - min_delta):
            best_mae = metrics["mae"]
            torch.save(model.state_dict(), best_path)
            stale_epochs = 0
        else:
            stale_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | Loss(MSE): {loss:.4f} | "
            f"Val MAE: {metrics['mae']:.4f} | Val RMSE: {metrics['rmse']:.4f} | "
            f"Rounded Acc: {metrics['acc_rounded']:.4f} | LR: {current_lr:.2e}"
        )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch:02d} (no MAE improvement for {patience} epochs).")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def train_epoch_classifier(model, loader, optimizer, criterion, device, class_weights=None):
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        x = batch["x"].to(device)
        tid = batch["task_ids"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y"].to(device).long()

        optimizer.zero_grad()
        logits, _ = model(x, tid, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        tid = batch["task_ids"].to(device)
        mask = batch["window_mask"].to(device)
        y = batch["y"].to(device).long()
        logits, _ = model(x, tid, mask)
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    if not all_targets:
        return {"loss": float("nan"), "accuracy": float("nan"), "f1_macro": float("nan"), "auc_macro": float("nan")}

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    accuracy = float(accuracy_score(targets, preds))
    f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
    try:
        auc_macro = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_macro = float("nan")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
    }


def fit_classifier(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=80,
    patience=12,
    min_delta=1e-3,
    scheduler_patience=5,
    scheduler_factor=0.5,
    min_lr=1e-5,
    class_weights=None,
    best_path="best_attention_classifier.pth",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    best_score = float("-inf")
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch_classifier(model, train_loader, optimizer, criterion, device, class_weights=class_weights)
        metrics = evaluate_classifier(model, val_loader, device)
        scheduler.step(metrics["f1_macro"])

        if metrics["f1_macro"] > (best_score + min_delta):
            best_score = metrics["f1_macro"]
            torch.save(model.state_dict(), best_path)
            stale_epochs = 0
        else:
            stale_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | Loss(CE): {loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | "
            f"Val F1: {metrics['f1_macro']:.4f} | Val AUC: {metrics['auc_macro']:.4f} | LR: {current_lr:.2e}"
        )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch:02d} (no F1 improvement for {patience} epochs).")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model