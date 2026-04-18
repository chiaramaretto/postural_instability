import torch
import torch.nn as nn

# --- STEP 1: Data Representation (DR-SAE) ---
class DR_SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1)
        ])
        self.dec_layers = nn.ModuleList([
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1)
        ])
        self.selu = nn.SELU()

    def forward(self, x):
        for enc in self.enc_layers:
            x = self.selu(enc(x))
        latent = x
        for i in range(len(self.dec_layers)-1, -1, -1):
            x = self.dec_layers[i](x)
            if i > 0: x = self.selu(x)
        return x, latent

class LFF_AE(nn.Module):
    def __init__(self, input_channels=6 * 256, c4_dim=256):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv1d(input_channels, 512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1, return_indices=True)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.conv4 = nn.Conv1d(128, c4_dim, kernel_size=3, padding=1)
        
        # Decoder
        self.deconv4 = nn.ConvTranspose1d(c4_dim, 128, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose1d(128, 256, kernel_size=3, padding=1)
        self.unpool2 = nn.MaxUnpool1d(kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(256, 512, kernel_size=3, padding=1)
        self.unpool1 = nn.MaxUnpool1d(kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose1d(512, input_channels, kernel_size=3, padding=1)
        
        self.selu = nn.SELU()
    
    def forward(self, x):
        # --- ENCODER ---
        x = self.selu(self.conv1(x))
        size1 = x.size()
        x, idx1 = self.pool1(x)

        x = self.selu(self.conv2(x))
        size2 = x.size()
        x, idx2 = self.pool2(x)

        x = self.selu(self.bn3(self.conv3(x)))
        latent = self.selu(self.conv4(x))

        # --- DECODER ---
        x = self.selu(self.deconv4(latent))
        x = self.selu(self.deconv3(x))

        x = self.unpool2(x, idx2, output_size=size2)
        x = self.selu(self.deconv2(x))

        x = self.unpool1(x, idx1, output_size=size1)
        recon = self.deconv1(x)

        return recon, latent
    


