import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import ConvAutoencoder
from train import train_model
import os

def main():
    # Setup
    if torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load data
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "..", "data", "windowed_data", "X_train.npy")

    X_raw = np.load(data_path)
    X_tensor = torch.from_numpy(X_raw).float()
    
    # Split Train/Val
    dataset = TensorDataset(X_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Train Autoencoder
    model = ConvAutoencoder().to(device)
    train_hist, val_hist = train_model(model, train_loader, val_loader, epochs=20, device=device)

    # Visualize
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Validation")
    plt.legend()
    plt.title("Loss Reconstruction")
    plt.show()


def plot_reconstruction(sample_idx=0):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "..", "data", "windowed_data", "X_train.npy")
    model_path = os.path.join(base_path, "checkpoint", "best_autoencoder.pth")
    # 1. Carica i dati e il modello
    if not os.path.exists(data_path):
        print(f"Errore: Non trovo i dati in {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"Errore: Non trovo il modello in {model_path}")
        return

    X_raw = np.load(data_path)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 2. Prepara il campione (Batch, Length, Channels)
    sample = torch.from_numpy(X_raw[sample_idx:sample_idx+1]).float().to(device)
    
    # 3. Passaggio nel modello
    with torch.no_grad():
        reconstructed, _ = model(sample)
    
    # Portiamo tutto in numpy per il plot
    orig = sample.cpu().squeeze().numpy()        # Shape: (256, 6)
    rec = reconstructed.cpu().squeeze().numpy()  # Shape: (256, 6)
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Sottogruppo Accelerometro (primi 3 canali)
    axes[0].plot(orig[:, :3], label=['Acc X', 'Acc Y', 'Acc Z'], alpha=0.4, linestyle='--')
    axes[0].set_prop_cycle(None) # Resetta i colori per farli coincidere
    axes[0].plot(rec[:, :3], label=['Rec X', 'Rec Y', 'Rec Z'], linewidth=2)
    axes[0].set_title(f"Accelerometro - Finestra {sample_idx}")
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Sottogruppo Giroscopio (ultimi 3 canali)
    axes[1].plot(orig[:, 3:], label=['Gyro X', 'Gyro Y', 'Gyro Z'], alpha=0.4, linestyle='--')
    axes[1].set_prop_cycle(None)
    axes[1].plot(rec[:, 3:], label=['Rec X', 'Rec Y', 'Rec Z'], linewidth=2)
    axes[1].set_title(f"Giroscopio - Finestra {sample_idx}")
    axes[1].legend(loc='upper right', ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    plot_reconstruction(sample_idx=100)