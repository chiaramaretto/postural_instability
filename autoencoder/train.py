import torch
import os
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in loader:
        inputs = batch[0].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss / len(loader.dataset)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    for batch in loader:
        inputs = batch[0].to(device)
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss / len(loader.dataset)

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu"):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_autoencoder.pth")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    pbar_epochs = tqdm(range(epochs), desc="Training Progress", unit="epoch")

    for epoch in pbar_epochs:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar_epochs.set_postfix(T_Loss=f"{train_loss:.4f}", V_Loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"  [Epoch {epoch+1}] New best model saved (Val Loss: {val_loss:.6f})")

    return train_losses, val_losses