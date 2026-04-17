from model import PosturalInstabilityClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm


def train_model(X_train, y_train, input_size=256, num_classes=5, batch_size=32, num_epochs=20, learning_rate=0.001):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function and optimizer
    model = PosturalInstabilityClassifier(input_size=input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return model