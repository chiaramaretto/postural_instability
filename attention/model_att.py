import torch
import torch.nn as nn
import torch.nn.functional as F

class SubjectTaskAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_tasks=3, dropout=0.3):
        super().__init__()
        self.num_tasks = num_tasks

        self.window_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        self.window_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.task_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Regression head: predicts postural_stability on a continuous scale.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _masked_softmax(self, logits, mask, dim):
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=dim)

    def forward(self, x, task_ids, window_mask):
        h = self.window_proj(x)
        task_vectors = []
        task_present = []

        for t in range(self.num_tasks):
            t_mask = (task_ids == t) & window_mask
            task_present.append(t_mask.any(dim=1))
            h_t = h + self.task_embedding.weight[t].view(1, 1, -1)
            attn_w = self._masked_softmax(self.window_attn(h_t).squeeze(-1), t_mask, dim=1)
            task_vectors.append(torch.bmm(attn_w.unsqueeze(1), h).squeeze(1))

        task_tensor = torch.stack(task_vectors, dim=1)
        task_mask = torch.stack(task_present, dim=1)
        t_attn_w = self._masked_softmax(self.task_attn(task_tensor).squeeze(-1), task_mask, dim=1)
        
        subject_embedding = torch.bmm(t_attn_w.unsqueeze(1), task_tensor).squeeze(1)
        pred = self.classifier(subject_embedding).squeeze(-1)
        
        return pred, {"task_weights": t_attn_w}


class SubjectTaskAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_tasks=3, num_classes=5, dropout=0.3):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes

        self.window_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        self.window_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.task_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _masked_softmax(self, logits, mask, dim):
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=dim)

    def forward(self, x, task_ids, window_mask):
        h = self.window_proj(x)
        task_vectors = []
        task_present = []

        for t in range(self.num_tasks):
            t_mask = (task_ids == t) & window_mask
            task_present.append(t_mask.any(dim=1))
            h_t = h + self.task_embedding.weight[t].view(1, 1, -1)
            attn_w = self._masked_softmax(self.window_attn(h_t).squeeze(-1), t_mask, dim=1)
            task_vectors.append(torch.bmm(attn_w.unsqueeze(1), h).squeeze(1))

        task_tensor = torch.stack(task_vectors, dim=1)
        task_mask = torch.stack(task_present, dim=1)
        t_attn_w = self._masked_softmax(self.task_attn(task_tensor).squeeze(-1), task_mask, dim=1)

        subject_embedding = torch.bmm(t_attn_w.unsqueeze(1), task_tensor).squeeze(1)
        logits = self.classifier(subject_embedding)

        return logits, {"task_weights": t_attn_w}