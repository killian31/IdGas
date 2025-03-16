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
[Fusion Layer]  ← Combining multi-head self-attention outputs
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

    def forward(self, x, return_attn=False):
        x = x.transpose(0, 1)  # (b, 13, 16) -> (13, b, 16) for multihead attention
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        if return_attn:
            return x, attn_weights
        return x


class FusionLayer(nn.Module):
    """
    Fusion layer for combining multi-head self-attention outputs.
    """

    def __init__(self, total_embed_dim, embed_dim, dropout_rate=0.1):
        super(FusionLayer, self).__init__()
        self.fc = nn.Linear(total_embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class RAMTEncoder(nn.Module):
    """
    Multi-head self-attention encoder for sequences.
    """

    def __init__(
        self,
        num_features,
        feature_embed_dim,
        embed_dim,
        num_heads,
        num_blocks,
        dropout_rate=0.1,
    ):
        super(RAMTEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedder = FeatureEmbedding(num_features, feature_embed_dim)
        self.attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(feature_embed_dim, num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.fusion_layer = FusionLayer(
            num_features * feature_embed_dim, embed_dim, dropout_rate
        )

    def forward(self, x):
        x = self.embedder(x)
        for block in self.attention_blocks:
            x = block(x)  # (b, 13, 16)
        x = x.flatten(1)  # (b, 13 * 16)
        x = self.fusion_layer(x)  # (b, 16)
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


class RAMTHead(nn.Module):
    """
    RAMTNet head module. Combines residual blocks and multi-task heads.
    """

    def __init__(
        self,
        in_features,
        res_hidden,
        num_tasks=23,
        num_res_blocks=2,
        head_hidden=64,
        dropout_rate=0.3,
    ):
        super(RAMTHead, self).__init__()
        self.res_blocks = nn.Sequential(
            nn.Linear(in_features, res_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[
                ResidualDenseBlock(res_hidden, res_hidden, dropout_rate)
                for _ in range(num_res_blocks)
            ],
        )
        self.multitask_head = MultiTaskHead(
            res_hidden, num_tasks, head_hidden, dropout_rate
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.multitask_head(x)
        return x


class RAMTNet(nn.Module):
    """
    Enhanced neural network for toxic gas detection using feature embeddings,
    self-attention, residual blocks, and multi-task heads.
    """

    def __init__(
        self,
        num_features=13,
        feature_embed_dim=8,
        embed_dim=16,
        attn_heads=4,
        attn_blocks=2,
        num_res_blocks=2,
        res_hidden=64,
        num_tasks=23,
        head_hidden=64,
        enc_dropout_rate=0.3,
        head_dropout_rate=0.3,
        **kwargs,
    ):
        super(RAMTNet, self).__init__()
        self.encoder = RAMTEncoder(
            num_features,
            feature_embed_dim,
            embed_dim,
            attn_heads,
            attn_blocks,
            enc_dropout_rate,
        )
        self.head = RAMTHead(
            embed_dim,
            res_hidden,
            num_tasks,
            num_res_blocks,
            head_hidden,
            head_dropout_rate,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
