import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import PosturalInstabilityClassifier
import tqdm

def train_model(X_train, y_train, input_size=256, num_classes=2, class_weights=None, batch_size=32, num_epochs=100, learning_rate=0.005):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PosturalInstabilityClassifier(input_size=input_size, num_classes=num_classes)
    
    # Se passiamo i pesi, la Loss darà più importanza alle classi meno numerose
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Binary Classifier"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    return model