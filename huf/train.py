import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_stacked_dr_sae(model, loader, device, lr=1e-3):

    model.to(device)
    criterion = nn.MSELoss()
    
    current_input_data = []
    for batch in loader:
        current_input_data.append(batch[0])
    current_input_data = torch.cat(current_input_data, dim=0)

    for i in range(len(model.enc_layers)):
        print(f"\n--- Training DR-SAE Layer {i+1}/5 ---")
        
        encoder_layer = model.enc_layers[i]
        decoder_layer = model.dec_layers[i]
        
        # reset optimizer
        optimizer = optim.Adam(list(encoder_layer.parameters()) + list(decoder_layer.parameters()), lr=lr)
        
        layer_loader = DataLoader(TensorDataset(current_input_data), batch_size=64, shuffle=True)
        
        epoch = 0
        pbar = tqdm(total=100, desc=f"Layer {i+1} Progress")
        
        while True:
            epoch += 1
            model.train()
            running_loss = 0.0
            
            for batch in layer_loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                
                z = model.selu(encoder_layer(inputs))
                reconstructed = decoder_layer(z)
                
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(layer_loader)
            
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})
            
            if epoch >= 5 and avg_loss < 0.005:
                pbar.write(f"Layer {i+1}, Epoch: {epoch}, Loss: {avg_loss:.6f}")
                pbar.close()
                break
            
            if epoch >= 100: 
                pbar.write(f"Layer {i+1} ha raggiunto il limite di salvaguardia (100 epoche).")
                pbar.close()
                break

        model.eval()
        with torch.no_grad():
            new_inputs = []
            eval_loader = DataLoader(TensorDataset(current_input_data), batch_size=64, shuffle=False)
            for batch in eval_loader:
                out = model.selu(encoder_layer(batch[0].to(device)))
                new_inputs.append(out.cpu())
            current_input_data = torch.cat(new_inputs, dim=0)
            
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
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(loader)
        epoch += 1
        
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