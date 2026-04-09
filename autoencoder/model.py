import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        # --- ENCODER (Basato su DR-SAE) ---
        self.encoder = nn.Sequential(
            # Input: (batch, 6, 256)
            nn.Conv1d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.SELU(), # [cite: 117, 337]
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SELU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SELU(),
            nn.BatchNorm1d(64), # [cite: 173]
            
            nn.Flatten(),
            nn.Linear(64 * 32, latent_dim),
            nn.Dropout(0.1) 
        )
        
        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, 64 * 32)
        
        self.decoder = nn.Sequential(
            # Torniamo su con kernel 3 come suggerito [cite: 176]
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
            nn.BatchNorm1d(32),
            
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
            nn.BatchNorm1d(16),
            
            nn.ConvTranspose1d(16, 6, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Nessuna attivazione finale per ricostruire i valori originali [cite: 140]
        )

    def forward(self, x):
        # Portiamo a (batch, 6, 256)
        x = x.permute(0, 2, 1)
        
        latent = self.encoder(x)
        
        x_rec = self.decoder_input(latent)
        x_rec = x_rec.view(-1, 64, 32)
        x_rec = self.decoder(x_rec)
        
        # Torniamo a (batch, 256, 6)
        x_rec = x_rec.permute(0, 2, 1)
        return x_rec, latent