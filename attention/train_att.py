import numpy as np
import torch
import torch.nn as nn

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