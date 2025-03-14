import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bias=True, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Dimension of the input features (i.e. the encoder's latent dimension).
            hidden_dim (int): Size of the first hidden layer.
            bias (bool): Whether to use bias in linear layers.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1, bias=bias),
            nn.Sigmoid(),  # Output is probability: 0 = source, 1 = target.
        )

    def forward(self, x):
        return self.net(x)
