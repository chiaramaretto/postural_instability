import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import tqdm
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def fit_model(model, X_train, y_train, device, batch_size=32, max_epochs=500, val_fraction=0.2, patience=20, X_val=None, y_val=None):
    os.makedirs('posturalInstability/cnn_gru/data/models', exist_ok=True)
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)

    if X_val is not None and y_val is not None:
        X_val = torch.as_tensor(X_val, dtype=torch.float32)
        y_val = torch.as_tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(X_val, y_val)
    else:
        val_size = max(1, int(len(train_dataset) * val_fraction)) if len(train_dataset) > 1 else 0
        train_size = len(train_dataset) - val_size
        if val_size == 0:
            train_dataset, val_dataset = train_dataset, train_dataset
        else:
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    epochs_without_improvement = 0
    best_state_dict = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': []
    }

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validation
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_losses.append(loss.item())
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        if len(all_targets) == 0:
            avg_val_loss = 0.0
            acc = 0.0
        else:
            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            try:
                acc = accuracy_score(all_targets, all_preds)
            except Exception:
                acc = 0.0

        # Compute other metrics
        try:
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        except Exception:
            f1 = 0.0
        try:
            precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        except Exception:
            precision = 0.0
        try:
            recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        except Exception:
            recall = 0.0

        # AUC (macro) if possible
        try:
            auc = roc_auc_score(np.eye(np.max(all_targets) + 1)[all_targets], np.array(all_probs), average='macro', multi_class='ovo')
        except Exception:
            auc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(acc)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_auc'].append(auc)

        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}, Val AUC: {auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, 'posturalInstability/cnn_gru/data/models/best_stacking_model.pt')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (patience={patience}).")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, best_acc, history


def predict(model, X, device, batch_size=32):
    X = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu())
    return torch.cat(predictions).numpy()


def print_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Final confusion matrix:")
    print(cm)
    return cm