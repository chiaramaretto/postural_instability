import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_stacked_dr_sae(
    model,
    loader,
    device,
    lr=1e-3,
    min_epochs=5,
    max_epochs=100,
    target_loss=0.005,
    layer_batch_size=32,
):

    model.to(device)
    criterion = nn.MSELoss()
    
    if hasattr(loader.dataset, "tensors") and len(loader.dataset.tensors) > 0:
        base_input_data = loader.dataset.tensors[0]
    else:
        cached_batches = [batch[0] for batch in loader]
        base_input_data = torch.cat(cached_batches, dim=0)

    for i in range(len(model.enc_layers)):
        print(f"\n--- Training DR-SAE Layer {i+1}/5 ---")
        
        encoder_layer = model.enc_layers[i]
        decoder_layer = model.dec_layers[i]
        
        # reset optimizer
        optimizer = optim.Adam(list(encoder_layer.parameters()) + list(decoder_layer.parameters()), lr=lr)
        
        layer_loader = DataLoader(
            TensorDataset(base_input_data),
            batch_size=layer_batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        
        epoch = 0
        pbar = tqdm(total=max_epochs, desc=f"Layer {i+1} Progress")
        
        while True:
            epoch += 1
            model.train()
            running_loss = 0.0
            
            for batch in layer_loader:
                inputs = batch[0].to(device, non_blocking=(device.type == "cuda"))

                if i > 0:
                    with torch.no_grad():
                        for j in range(i):
                            inputs = model.selu(model.enc_layers[j](inputs))

                optimizer.zero_grad()
                
                z = model.selu(encoder_layer(inputs))
                reconstructed = decoder_layer(z)
                
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(layer_loader)
            
            pbar.update(1)
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})
            
            if epoch >= min_epochs and avg_loss < target_loss:
                pbar.write(f"Layer {i+1}, Epoch: {epoch}, Loss: {avg_loss:.6f}")
                pbar.close()
                break
            
            if epoch >= max_epochs:
                pbar.write(f"Layer {i+1} ha raggiunto il limite di salvaguardia (100 epoche).")
                pbar.close()
                break
            
    return model

def train_fusion_block(model, loader, device, lr=1e-3, block_name="Fusion"):
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\n--- Training {block_name} Block ---")
    epoch = 0
    pbar = tqdm(total=100, desc=f"{block_name} Training")
    
    while True:
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = batch[0].to(device, non_blocking=(device.type == "cuda")) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad()
            
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(loader)
        epoch += 1
        
        pbar.update(1)
        pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})
        
        if epoch >= 5 and avg_loss < 0.005:
            pbar.write(f"{block_name} Target raggiunto! Epoche: {epoch}, Loss Finale: {avg_loss:.6f}")
            pbar.close()
            break
        
        if epoch >= 100: 
            pbar.write(f"{block_name} ha raggiunto il limite di salvaguardia (100 epoche).")
            pbar.close()
            break
            
    return model