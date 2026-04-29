import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import tqdm
import os

def fit_model(model, X_train, y_train, device, batch_size=32, max_epochs=100, val_fraction=0.2):
    os.makedirs('posturalInstability/cnn_gru/data/models', exist_ok=True)
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        # at each iteration print loss and accuracy on the training set
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss: {loss.item():.4f}", end="\r")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{max_epochs}, Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'posturalInstability/cnn_gru/data/models/best_stacking_model.pt')
            pbar.set_postfix({"Best Acc": best_acc})
    return model, best_acc