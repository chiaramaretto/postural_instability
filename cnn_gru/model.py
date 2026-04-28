import torch
import torch.nn as nn


class GRUCNNEnsembleClassifier(nn.Module):
    """Stacking ensemble: two 1D-CNN heads (kernel 1 and 3) and a GRU head.

    Outputs logits for 5 classes.
    """

    def __init__(self, input_channels: int = 6, n_filters: int = 64, gru_hidden: int = 64, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.input_channels = input_channels

        # 1D-CNN head kernel size 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=n_filters, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
        )

        # 1D-CNN head kernel size 3
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
        )

        # Project raw channels to GRU input size (time-distributed GRU input)
        self.project = nn.Conv1d(in_channels=input_channels, out_channels=n_filters, kernel_size=1)
        self.gru = nn.GRU(input_size=n_filters, hidden_size=gru_hidden, batch_first=True)

        # Apply adaptive pooling to keep CNN outputs fixed-size
        self.cnn1_pool = nn.AdaptiveAvgPool1d(8)
        self.cnn3_pool = nn.AdaptiveAvgPool1d(8)

        fused_size = n_filters * 8 + n_filters * 8 + gru_hidden
        self.meta = nn.Sequential(
            nn.Linear(fused_size, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        if x.ndim != 3:
            raise ValueError("Expected 3D input (batch, seq_len, channels)")

        # ensure (batch, channels, seq_len)
        if x.shape[1] > x.shape[2]:
            x_cf = x
        else:
            x_cf = x.permute(0, 2, 1)

        # CNN heads operate on channels-first
        p1 = self.cnn1_pool(x_cf).reshape(x_cf.size(0), -1)
        p3 = self.cnn3_pool(x_cf).reshape(x_cf.size(0), -1)

        # GRU head
        proj = self.project(x_cf)  # (batch, n_filters, seq_len)
        proj = proj.permute(0, 2, 1)  # (batch, seq_len, n_filters)
        _, h = self.gru(proj)
        gru_feat = h[-1]

        fused = torch.cat([p1, p3, gru_feat], dim=1)
        logits = self.meta(fused)
        return logits
