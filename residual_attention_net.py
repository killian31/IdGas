import torch
import torch.nn as nn

"""
Residual Attention Multi-Task Network (RAMTNet)

Inputs (13 features)
    │
[Input Embedding & Layer Norm]  ← Embedding each sensor reading to a higher dimension
    │
[Multi-head Self-Attention Layer + Residual Connection]
    │
[Residual Dense Block 1]  ← FC → BN → ReLU → FC → BN + Skip
    │
[Residual Dense Block 2]  ← (Optional additional block)
    │
[Shared Fully Connected Layer]
    │
┌───────────────┬───────────────┬──────────── ... ───────────────┐
│  Head 1       │  Head 2       │   ...            │  Head 23    │
│ (Dense + Sigmoid)   (Dense + Sigmoid)           (Dense + Sigmoid)
└───────────────┴───────────────┴──────────── ... ─┘─────────────┘

"""


class FeatureEmbedding(nn.Module):
    """Embeds each input feature independently using a linear layer."""

    def __init__(self, num_features=13, embed_dim=16):
        super(FeatureEmbedding, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_features)]
        )

    def forward(self, x):
        embedded = [
            self.embeddings[i](x[:, i : i + 1]) for i in range(self.num_features)
        ]
        return torch.stack(embedded, dim=1)


class SelfAttentionBlock(nn.Module):
    """
    Applies multi-head self-attention with residual connection and layer normalization.
    """

    def __init__(self, embed_dim=16, num_heads=4, dropout_rate=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(0, 1)  # (b, 13, 16) -> (13, b, 16) for multihead attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        return x


class ResidualDenseBlock(nn.Module):
    """
    A residual block with two fully connected layers, dropout and batch normalization.
    """

    def __init__(self, features, hidden_features, dropout_rate=0.3):
        super(ResidualDenseBlock, self).__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class MultiTaskHead(nn.Module):
    """Multi-task head with separate sub-networks for each target."""

    def __init__(self, in_features, num_tasks=23, head_hidden=64, dropout_rate=0.3):
        super(MultiTaskHead, self).__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(head_hidden, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=1)


class RAMTNet(nn.Module):
    """
    Enhanced neural network for toxic gas detection using feature embeddings,
    self-attention, residual blocks, and multi-task heads.
    """

    def __init__(
        self,
        num_features=13,
        embed_dim=16,
        num_heads=4,
        shared_dim=128,
        num_tasks=23,
        num_res_blocks=2,
        dropout_rate=0.3,
        head_hidden=64,
    ):
        super(RAMTNet, self).__init__()
        self.embedding = FeatureEmbedding(num_features, embed_dim)
        self.attention = SelfAttentionBlock(embed_dim, num_heads, dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared_fc = nn.Linear(embed_dim, shared_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.res_blocks = nn.Sequential(
            *[
                ResidualDenseBlock(shared_dim, shared_dim, dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )
        self.multitask_head = MultiTaskHead(
            shared_dim, num_tasks, head_hidden, dropout_rate
        )

    def forward(self, x):
        x = self.embedding(x)  # (b, 13, 16)
        x = self.attention(x)  # (b, 13, 16)
        x = x.transpose(1, 2)  # (b, 16, 13)
        x = self.pool(x).squeeze(-1)  # (b, 16) pool over the sequence dimension
        x = self.shared_fc(x)  # (b, 128)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_blocks(x)  # (b, 128)
        x = self.multitask_head(x)  # (b, 23)
        return x
