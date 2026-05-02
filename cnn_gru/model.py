import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnGru(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, batch_size=32):
        super(CnnGru, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.cnn1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(50),
            nn.Flatten()
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(50),
            nn.Flatten()
        )
        
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )


        self.dense_layer = nn.Sequential(
            # Infer input features automatically at first forward pass.
            nn.LazyLinear(100), 
            nn.ReLU(),
            nn.Dropout(0.5),      
            nn.Linear(100, num_classes)     
        )

    def forward(self, x):
        
        # Conv1d expects (batch, channels, length)
        x_cnn = x.transpose(1, 2)
        c1 = self.cnn1(x_cnn)
        c2 = self.cnn2(x_cnn)

        # GRU expects (batch, time, channels); use temporal mean to keep richer sequence info.
        gru_out, _ = self.gru(x)
        g = gru_out.mean(dim=1)

        x = torch.cat((c1, c2, g), dim=1)
        logits = self.dense_layer(x)
        return logits
     