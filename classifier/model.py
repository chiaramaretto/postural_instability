import torch.nn as nn

class PosturalInstabilityClassifier(nn.Module):
    def __init__(self, input_size=256, num_classes=2):
        super(PosturalInstabilityClassifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3) 
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc_out(out)
        return out