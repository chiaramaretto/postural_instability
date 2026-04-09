import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_stacked_dr_sae(model, loader, device, lr=1e-3):
    """
    Allena il DR-SAE strato dopo strato (Greedy Layer-wise Training) come descritto nel paper.
    Questo metodo decompone il segnale in diverse scale[cite: 102, 103].
    """
    model.to(device)
    criterion = nn.MSELoss()
    
    # Raccogliamo i dati grezzi iniziali per il primo strato
    current_input_data = []
    for batch in loader:
        current_input_data.append(batch[0])
    current_input_data = torch.cat(current_input_data, dim=0)

    # Ciclo sui 5 livelli del DR-SAE [cite: 100, 115]
    for i in range(len(model.enc_layers)):
        print(f"\n--- Training DR-SAE Layer {i+1}/5 ---")
        
        encoder_layer = model.enc_layers[i]
        decoder_layer = model.dec_layers[i]
        
        # Reset dell'ottimizzatore per ogni strato per forzare l'apprendimento di feature significative [cite: 106]
        optimizer = optim.Adam(list(encoder_layer.parameters()) + 
                               list(decoder_layer.parameters()), lr=lr)
        
        layer_loader = DataLoader(TensorDataset(current_input_data), batch_size=64, shuffle=True)
        
        epoch = 0
        # Barra di progresso tqdm per monitorare la loss in tempo reale
        pbar = tqdm(total=100, desc=f"Layer {i+1} Progress")
        
        while True:
            epoch += 1
            model.train()
            running_loss = 0.0
            
            for batch in layer_loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                
                # Forward locale: Encoder -> SELU -> Decoder
                z = model.selu(encoder_layer(inputs))
                reconstructed = decoder_layer(z)
                
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(layer_loader)
            
            # Aggiornamento della barra con la loss corrente
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{avg_loss:.6f}"})
            
            # Criterio del paper: Loss < 0.005 e almeno 5 epoche [cite: 341]
            if epoch >= 5 and avg_loss < 0.005:
                pbar.write(f"Layer {i+1} Target raggiunto! Epoche: {epoch}, Loss Finale: {avg_loss:.6f}")
                pbar.close()
                break
            
            if epoch >= 100: 
                pbar.write(f"Layer {i+1} ha raggiunto il limite di salvaguardia (100 epoche).")
                pbar.close()
                break

        # Passaggio dei dati attraverso lo strato appena addestrato per alimentare il successivo [cite: 102]
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
    """
    Training per i blocchi di fusione LFF e GFF. 
    Questi blocchi unificano le feature dei sensori in un unico set[cite: 9, 36, 197].
    """
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
        
        # Applichiamo lo stesso criterio di stabilità per coerenza con il DR-SAE
        if epoch >= 5 and avg_loss < 0.005:
            pbar.write(f"{block_name} Target raggiunto! Epoche: {epoch}, Loss Finale: {avg_loss:.6f}")
            pbar.close()
            break
        
        if epoch >= 100: 
            pbar.write(f"{block_name} ha raggiunto il limite di salvaguardia (100 epoche).")
            pbar.close()
            break
            
    return model