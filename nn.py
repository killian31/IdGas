import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


from residual_attention_net import RAMTNet
from uda import GradientReversal, DomainDiscriminator
from utils import write_submissions, WarmupScheduler


class StackEnsemble(nn.Module):
    def __init__(self, models, groups):
        """
        models: list of models
        groups: list of tuples with lower and upper bounds of the group
        """
        super(StackEnsemble, self).__init__()
        self.models = models
        self.groups = groups
        self.fc = nn.Linear(13, 1)  # dummy layer for parameters init

    def forward(self, x):
        x_groups = []
        for i in range(len(self.groups)):
            x_groups.append(
                x[(x[:, 0] > self.groups[i][0]) & (x[:, 0] <= self.groups[i][1])]
            )
        # then pass each group through its corresponding model
        y = []
        for i in range(len(self.models)):
            if len(x_groups[i]) == 0:
                continue
            y.append(self.models[i](x_groups[i]))
        y = torch.cat(y, dim=0)
        return y


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, bias=True):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim, bias=bias),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiModalEncoder(nn.Module):
    def __init__(self, humidity_dim, group_embed_dim, bias=True):
        super(MultiModalEncoder, self).__init__()
        self.hum_mlp = MLPBlock(1, humidity_dim, bias=bias)
        self.m12_m15_mlp = MLPBlock(4, group_embed_dim, bias=bias)
        self.m4_7_mlp = MLPBlock(4, group_embed_dim, bias=bias)
        self.rs_mlp = MLPBlock(4, group_embed_dim, bias=bias)
        self.embed_dim = humidity_dim + group_embed_dim * 3

    def forward(self, x):
        """
        Expects x to be (batch_size, 13) with order:
         x[:,0]   = humidity
         x[:,1:5] = M12, M13, M14, M15
         x[:,5:9] = M4,  M5,  M6,  M7
         x[:,9:13]= R,   S1,  S2,  S3
        """
        # 1) Separate humidity from the sensor groups
        humidity = x[:, 0:1]
        m12_15 = x[:, 1:5]
        m4_7 = x[:, 5:9]
        rs = x[:, 9:13]
        # 2) Forward through each small MLP
        hum_embed = self.hum_mlp(humidity)
        m12_15_embed = self.m12_m15_mlp(m12_15)
        m4_7_embed = self.m4_7_mlp(m4_7)
        rs_embed = self.rs_mlp(rs)
        # 3) Concatenate all embeddings
        fused = torch.cat([hum_embed, m12_15_embed, m4_7_embed, rs_embed], dim=1)
        return fused


class RegressorHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_output, bias=True, dropout_rate=0.3):
        super(RegressorHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=bias),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_output, bias=bias),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)


class RegressorV3(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_output=23,
        humidity_dim=16,
        group_embed_dim=16,
        bias=True,
        dropout_rate=0.3,
    ):
        """
        Args:
          hidden_dim (int): Size of the hidden dimension for the final regressor MLP.
          num_output (int): Number of gas categories, i.e. c1–c23.
          humidity_dim (int): Size of the hidden dimension in the humidity MLP.
          group_embed_dim (int): Size of the hidden dimension in each sensor group's MLP.
          bias (bool): Whether to use bias in linear layers.
          dropout_rate (float): Dropout probability for regularization.
        """
        super(RegressorV3, self).__init__()

        self.encoder = MultiModalEncoder(
            humidity_dim=humidity_dim,
            group_embed_dim=group_embed_dim,
            bias=bias,
        )

        self.head = RegressorHead(
            input_dim=self.encoder.embed_dim,
            hidden_dim=hidden_dim,
            num_output=num_output,
            bias=bias,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Expects x to be (batch_size, 13) with order:
         x[:,0]   = humidity
         x[:,1:5] = M12, M13, M14, M15
         x[:,5:9] = M4,  M5,  M6,  M7
         x[:,9:13]= R,   S1,  S2,  S3
        """
        fused = self.encoder(x)
        out = self.head(fused)
        return out


class NNRegressor(nn.Module):
    def __init__(
        self, hidden_dim=128, num_output=23, bias=True, dropout_rate=0.3, **kwargs
    ):
        super(NNRegressor, self).__init__()
        self.fc_M10 = nn.Linear(5, 4, bias=bias)
        self.bn_M10 = nn.BatchNorm1d(4)
        self.fc_M0 = nn.Linear(5, 4, bias=bias)
        self.bn_M0 = nn.BatchNorm1d(4)
        self.fc_rs = nn.Linear(5, 8, bias=bias)
        self.bn_rs = nn.BatchNorm1d(8)
        self.hidden1 = nn.Linear(16, hidden_dim, bias=bias)
        self.bn_hidden1 = nn.BatchNorm1d(hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=bias)
        self.bn_hidden2 = nn.BatchNorm1d(hidden_dim // 2)
        self.regressor = nn.Linear(hidden_dim // 2, num_output, bias=bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_M10 = x[:, :5]  # humidity and M12 to M15
        x_M0 = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 5:9]), dim=1
        )  # humidity and M4 to M7
        x_rs = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 9:13]), dim=1
        )  # humidity and R S1 to S3

        x_M10 = self.bn_M10(self.relu(self.fc_M10(x_M10)))
        x_M0 = self.bn_M0(self.relu(self.fc_M0(x_M0)))
        x_rs = self.bn_rs(self.relu(self.fc_rs(x_rs)))

        x = torch.cat((x_M10, x_M0, x_rs), dim=1)
        x = self.bn_hidden1(self.relu(self.hidden1(x)))
        x = self.dropout(x)
        x = self.bn_hidden2(self.relu(self.hidden2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.regressor(x))

        return x


class GasDetectionModel(nn.Module):
    def __init__(self, hidden_dim=64, num_outputs=23, dropout_rate=0.2, **kwargs):
        super(GasDetectionModel, self).__init__()

        # Humidity-aware processing
        self.humidity_encoder = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU()
        )

        # Sensor group specific processing with humidity conditioning
        # M12-M15 group
        self.m12_15_encoder = nn.Sequential(
            nn.Linear(4, 12), nn.ReLU(), nn.BatchNorm1d(12)
        )
        self.m12_15_cond = nn.Linear(8, 12)  # Humidity conditioning

        # M4-M7 group
        self.m4_7_encoder = nn.Sequential(
            nn.Linear(4, 12), nn.ReLU(), nn.BatchNorm1d(12)
        )
        self.m4_7_cond = nn.Linear(8, 12)  # Humidity conditioning

        # R, S1-S3 group
        self.rs_encoder = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.BatchNorm1d(12))
        self.rs_cond = nn.Linear(8, 12)  # Humidity conditioning

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
        )

        # Multiple output heads for different gas types
        # For robustness, we create separate output heads for different gas categories
        self.output_head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        # Extract humidity and normalize it to enhance generalization
        humidity = x[:, 0:1]  # Shape: [batch_size, 1]

        # Extract sensor groups
        m12_15 = x[:, 1:5]  # M12, M13, M14, M15
        m4_7 = x[:, 5:9]  # M4, M5, M6, M7
        rs = x[:, 9:13]  # R, S1, S2, S3

        # Process humidity as conditioning information
        h_encoding = self.humidity_encoder(humidity)

        # Process each sensor group with humidity conditioning
        m12_15_feat = self.m12_15_encoder(m12_15)
        m12_15_feat = m12_15_feat * torch.sigmoid(self.m12_15_cond(h_encoding))

        m4_7_feat = self.m4_7_encoder(m4_7)
        m4_7_feat = m4_7_feat * torch.sigmoid(self.m4_7_cond(h_encoding))

        rs_feat = self.rs_encoder(rs)
        rs_feat = rs_feat * torch.sigmoid(self.rs_cond(h_encoding))

        # Combine features
        combined_features = torch.cat([m12_15_feat, m4_7_feat, rs_feat], dim=1)

        # Process combined features
        features = self.combined(combined_features)

        # Generate outputs
        outputs = torch.sigmoid(self.output_head(features))

        return outputs


class GatedSensorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Maps raw sensor values to latent
        self.lin_sensors = nn.Linear(in_features, out_features, bias=True)
        # Gate that modulates by humidity
        self.lin_humidity = nn.Linear(1, out_features, bias=False)
        self.bn_out = nn.BatchNorm1d(out_features)

    def forward(self, sensors, humidity):
        # sensors: (batch, in_features)
        # humidity: (batch, 1)
        sensor_embed = self.lin_sensors(sensors)
        hum_gate = self.lin_humidity(humidity)
        # Weighted combination
        combined = sensor_embed + hum_gate
        return self.bn_out(F.relu(combined))


class NNRegressorV2(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3, **kwargs):
        super().__init__()
        # M4–M7 block
        self.block_M0 = GatedSensorBlock(in_features=4, out_features=16)
        # M12–M15 block
        self.block_M10 = GatedSensorBlock(in_features=4, out_features=16)
        # R, S1, S2, S3 block
        self.block_RS = GatedSensorBlock(in_features=4, out_features=16)

        # After gating
        self.fc_fuse1 = nn.Linear(16 * 3, hidden_dim)
        self.bn_fuse1 = nn.BatchNorm1d(hidden_dim)
        self.fc_fuse2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fuse2 = nn.BatchNorm1d(hidden_dim // 2)
        self.regressor = nn.Linear(hidden_dim // 2, 23)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x shape: (batch_size, 13)
        Let's define:
         x[:,0] = Humidity
         x[:,1:5] = M12, M13, M14, M15
         x[:,5:9] = M4, M5, M6, M7
         x[:,9:13] = R, S1, S2, S3
        """
        humidity = x[:, 0:1]
        M12_15 = x[:, 1:5]
        M4_7 = x[:, 5:9]
        R_S1S2S3 = x[:, 9:13]

        out_M0 = self.block_M0(M4_7, humidity)
        out_M10 = self.block_M10(M12_15, humidity)
        out_RS = self.block_RS(R_S1S2S3, humidity)

        fused = torch.cat([out_M0, out_M10, out_RS], dim=1)
        fused = self.bn_fuse1(F.relu(self.fc_fuse1(fused)))
        fused = self.dropout(fused)
        fused = self.bn_fuse2(F.relu(self.fc_fuse2(fused)))
        fused = self.dropout(fused)

        out = torch.sigmoid(self.regressor(fused))
        return out

    def embed(self, x):
        humidity = x[:, 0:1]
        M12_15 = x[:, 1:5]
        M4_7 = x[:, 5:9]
        R_S1S2S3 = x[:, 9:13]

        out_M0 = self.block_M0(M4_7, humidity)
        out_M10 = self.block_M10(M12_15, humidity)
        out_RS = self.block_RS(R_S1S2S3, humidity)

        fused = torch.cat([out_M0, out_M10, out_RS], dim=1)

        fused = self.bn_fuse1(F.relu(self.fc_fuse1(fused)))

        return fused


class BasicDeepRegressor(nn.Module):
    """
    Basic network that takes a k-dimensional input features vector and
    outputs multiple values.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        dropout_rate,
        activation="relu",
        activation_out="sigmoid",
    ):
        super(BasicDeepRegressor, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(f"Activation '{activation}' not implemented")
        if activation_out == "sigmoid":
            self.activation_out = nn.Sigmoid()
        elif activation_out == "none":
            self.activation_out = nn.Identity()
        else:
            raise NotImplementedError(f"Activation '{activation_out}' not implemented")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(self.activation)
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(self.activation)
        # Output layer
        self.regressor = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        x = self.regressor(x)
        x = self.activation_out(x)
        return x


class SensorCNN(nn.Module):
    def __init__(
        self,
        input_dim=13,
        embed_dim=32,
        conv_channels=64,
        final_dim=32,
        num_output=23,
        dropout_rate=0.3,
    ):
        super(SensorCNN, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_in = nn.Linear(input_dim, embed_dim)

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=conv_channels // 8, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(conv_channels // 8)
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels // 8,
            out_channels=conv_channels // 4,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(conv_channels // 4)
        self.conv3 = nn.Conv1d(
            in_channels=conv_channels // 4,
            out_channels=conv_channels // 2,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(conv_channels // 2)
        self.conv4 = nn.Conv1d(
            in_channels=conv_channels // 2,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn4 = nn.BatchNorm1d(conv_channels)

        self.fusion_fc = nn.Linear(conv_channels * embed_dim, final_dim)

        self.output_layer = nn.Linear(final_dim, num_output)

    def forward(self, x):
        """
        x: (batch_size, 13)  ->  linear_in -> (batch_size, embed_dim)
        Then reshape for conv: (batch_size, 1, embed_dim)
        """
        x = self.linear_in(x)
        x = F.relu(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fusion_fc(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output_layer(x)
        return x


class WeightedRMSELoss(nn.Module):
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, output, target):
        weights = torch.where(target < 0.5, 1.0, 1.2)
        squared_errors = weights * (output - target) ** 2
        sample_errors = torch.mean(squared_errors, dim=1)
        loss = torch.sqrt(torch.mean(sample_errors))
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, output, target):
        weights = torch.where(target < 0.5, 1.0, 1.2)
        squared_errors = weights * (output - target) ** 2
        loss = torch.mean(squared_errors)
        return loss


class ZeroInflatedWeightedRMSELoss(nn.Module):
    def __init__(self, zero_penalty_factor=2.0):
        super().__init__()
        self.zero_penalty_factor = zero_penalty_factor

    def forward(self, output, target):
        weights = torch.where(target < 0.5, 1.0, 1.2)
        squared_error = weights * (output - target) ** 2

        zero_penalty = (target == 0).float() * (output**2)

        combined_loss = squared_error + zero_penalty * self.zero_penalty_factor

        mean_loss = torch.mean(squared_error + zero_penalty)
        return torch.sqrt(mean_loss)


def plot_loss(train_losses, val_losses, labels):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training")
    for val_loss, label in zip(val_losses, labels):
        plt.plot(val_loss, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train_nn_regressor(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def evaluate_nn_regressor(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    if isinstance(criterion, WeightedMSELoss):
        val_loss = val_loss**0.5
    return val_loss


def train_nn_regressor_uda(
    model,
    domain_disc,
    grl,
    source_loader,
    target_loader,
    optimizer,
    optimizer_disc,
    criterion,
    domain_criterion,
    device,
    epoch,
    warmup_epochs,
):
    model.train()
    domain_disc.train()
    disc_loss = 0
    reg_loss = 0
    train_loss = 0
    disc_correct = 0
    total = 0
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    # Loop through source batches
    for i in range(len(source_loader)):
        try:
            source_data, source_target = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_data, source_target = next(source_iter)

        if epoch >= warmup_epochs:
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            target_data = target_data.to(device)
        source_data, source_target = source_data.to(device), source_target.to(device)

        # --- Forward pass for source samples ---
        # Obtain latent features using the encoder (inside RegressorV3)
        fused_source = model.encoder(source_data)
        preds = model.head(fused_source)
        loss_reg = criterion(preds, source_target)

        # --- Domain Discriminator Loss ---
        # Get latent features for target samples
        if epoch >= warmup_epochs:
            fused_target = model.encoder(target_data)
            # Combine source and target latent features
            combined = torch.cat([fused_source, fused_target], dim=0)
            # Domain labels: 0 for source, 1 for target
            domain_labels = torch.cat(
                [torch.zeros(fused_source.size(0)), torch.ones(fused_target.size(0))],
                dim=0,
            ).to(device)
            # Pass through GRL then domain discriminator
            reversed_features = grl(combined)
            domain_preds = domain_disc(reversed_features)
            loss_domain = domain_criterion(domain_preds.view(-1), domain_labels)

            # --- Total Loss ---
            total_loss = loss_reg + loss_domain
        else:
            total_loss = loss_reg
        optimizer.zero_grad()
        if epoch >= warmup_epochs:
            optimizer_disc.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch >= warmup_epochs:
            optimizer_disc.step()

        train_loss += total_loss.item()
        if epoch >= warmup_epochs:
            with torch.no_grad():
                disc_loss += loss_domain.item()
                reg_loss += loss_reg.item()
                predicted = (domain_preds.view(-1) > 0.5).float()
                total += domain_labels.size(0)
                disc_correct += (predicted == domain_labels.view(-1)).sum().item()
    return (
        train_loss / len(source_loader),
        disc_loss / len(source_loader),
        (
            reg_loss / len(source_loader)
            if epoch >= warmup_epochs
            else train_loss / len(source_loader)
        ),
        disc_correct / total if epoch >= warmup_epochs else 0,
    )


def instantiate_model(model_class, model_params):
    if model_class == "NNRegressor":
        model = NNRegressor(**model_params)
    elif model_class == "GasDetectionModel":
        model = GasDetectionModel(**model_params)
    elif model_class == "NNRegressorV2":
        model = NNRegressorV2(**model_params)
    elif model_class == "BasicDeepRegressor":
        model = BasicDeepRegressor(**model_params)
    elif model_class == "RAMTNet":
        model = RAMTNet(**model_params)
    elif model_class == "RegressorV3":
        model = RegressorV3(**model_params)
    elif model_class == "SensorCNN":
        model = SensorCNN(**model_params)
    else:
        raise NotImplementedError(f"Model class '{model_class}' not implemented")
    return model


def compute_embeddings(model, x_train=None, x_val=None, x_test=None):
    if x_train is None and x_val is None and x_test is None:
        return None
    model.eval()
    with torch.no_grad():
        train_embeddings = (
            model.encoder(
                torch.tensor(x_train.drop("ID", axis=1).values, dtype=torch.float32)
            )
            if x_train is not None
            else None
        )
        val_embeddings = (
            model.encoder(
                torch.tensor(x_val.drop("ID", axis=1).values, dtype=torch.float32)
            )
            if x_val is not None
            else None
        )
        test_embeddings = (
            model.encoder(
                torch.tensor(x_test.drop("ID", axis=1).values, dtype=torch.float32)
            )
            if x_test is not None
            else None
        )

    return train_embeddings, val_embeddings, test_embeddings


def compare_active_dimensions(
    train_embeddings=None, val_embeddings=None, test_embeddings=None
):
    if train_embeddings is None and val_embeddings is None and test_embeddings is None:
        return None
    train_nonzero = (
        (train_embeddings != 0).sum(dim=1).float()
        if train_embeddings is not None
        else None
    )
    val_nonzero = (
        (val_embeddings != 0).sum(dim=1).float() if val_embeddings is not None else None
    )
    test_nonzero = (
        (test_embeddings != 0).sum(dim=1).float()
        if test_embeddings is not None
        else None
    )

    train_mean_active = (
        train_nonzero.mean().item() if train_nonzero is not None else None
    )
    val_mean_active = val_nonzero.mean().item() if val_nonzero is not None else None
    test_mean_active = test_nonzero.mean().item() if test_nonzero is not None else None

    embed_dim = (
        train_embeddings.shape[1]
        if train_embeddings is not None
        else (
            val_embeddings.shape[1]
            if val_embeddings is not None
            else test_embeddings.shape[1]
        )
    )

    print(f"Embedding dimension: {embed_dim}")
    (
        print(
            f"Average active dimensions in train: {train_mean_active:.2f} ({train_mean_active/embed_dim*100:.2f}%)"
        )
        if train_mean_active is not None
        else None
    )
    (
        print(
            f"Average active dimensions in val: {val_mean_active:.2f} ({val_mean_active/embed_dim*100:.2f}%)"
        )
        if val_mean_active is not None
        else None
    )
    (
        print(
            f"Average active dimensions in test: {test_mean_active:.2f} ({test_mean_active/embed_dim*100:.2f}%)"
        )
        if test_mean_active is not None
        else None
    )
    print("---")
    (
        print(f"Max active dimensions in train: {train_nonzero.max().item()}")
        if train_nonzero is not None
        else None
    )
    (
        print(f"Max active dimensions in val: {val_nonzero.max().item()}")
        if val_nonzero is not None
        else None
    )
    (
        print(f"Max active dimensions in test: {test_nonzero.max().item()}")
        if test_nonzero is not None
        else None
    )
    plt.figure(figsize=(10, 6))

    if train_nonzero is not None:
        counts, bins = np.histogram(
            train_nonzero.cpu().numpy(), bins=int(train_nonzero.max().item())
        )
        percentages = counts / counts.sum()
        plt.bar(
            bins[:-1],
            percentages,
            width=np.diff(bins),
            alpha=0.5,
            label="Train",
            align="edge",
        )

    if val_nonzero is not None:
        counts, bins = np.histogram(
            val_nonzero.cpu().numpy(), bins=int(val_nonzero.max().item())
        )
        percentages = counts / counts.sum()
        plt.bar(
            bins[:-1],
            percentages,
            width=np.diff(bins),
            alpha=0.3,
            label="Val",
            align="edge",
        )

    if test_nonzero is not None:
        counts, bins = np.histogram(
            test_nonzero.cpu().numpy(), bins=int(test_nonzero.max().item())
        )
        percentages = counts / counts.sum()
        plt.bar(
            bins[:-1],
            percentages,
            width=np.diff(bins),
            alpha=0.5,
            label="Test",
            align="edge",
        )

    plt.xlabel("Number of Active Dimensions")
    plt.ylabel("Proportion")
    plt.title("Distribution of Active Dimensions in Embeddings")
    plt.legend()
    plt.show()


def compute_lambda(p, gamma=10):
    """
    Compute lambda for GRL.
    """
    return 2 / (1 + np.exp(-gamma * p)) - 1


def run_experiment(
    x_source_train,
    y_source_train,
    x_target_train,
    valsets,
    params=None,
    uda=False,
    uda_hidden_dim=64,
    uda_dropout_rate=0.0,
    gamma=10,
    uda_phase_lr=1e-5,
    uda_disc_lr=1e-3,
    warmup_epochs=20,
    verbose=False,
    plot_losses=False,
    labels=None,
    zero_weight=2.0,
):
    """
    x_source_train, y_source_train: source training data and labels (pandas.DataFrame).
    x_target_train: target training data (unlabeled).
    valsets: list of (x_val, y_val) tuples for validation.
    params: dictionary of parameters.
    uda: whether to use unsupervised domain adaptation.
    uda_hidden_dim: hidden dimension for domain discriminator.
    uda_dropout_rate: dropout rate for domain discriminator.
    gamma: parameter for lambda annealing in GRL.
    uda_phase_lr: learning rate for the encoder during domain adaptation phase.
    uda_disc_lr: learning rate for the domain discriminator.
    warmup_epochs: number of epochs to warm up the encoder before domain adaptation.
    verbose: whether to print training progress.
    plot_losses: whether to plot training and validation losses.
    labels: list of labels for validation sets.
    zero_weight: weighting factor for zero loss.
    """
    if params is None:
        params = {
            "model_class": "RegressorV3",
            "model_params": {
                "hidden_dim": 128,
                "num_output": 23,
                "humidity_dim": 16,
                "group_embed_dim": 16,
                "bias": True,
                "dropout_rate": 0.3,
            },
            "training_params": {
                "n_epochs": 100,
                "lr": 0.001,
                "weight_decay": 1e-5,
                "batch_size": 128,
                "patience": 5,
                "factor": 0.5,
                "min_lr": 1e-6,
                "lr_warmup_epochs": 1,
                "loss": "WeightedRMSELoss",
            },
        }

    model_params = params["model_params"]
    training_params = params["training_params"]

    # Instantiate model based on model_class
    model = instantiate_model(params["model_class"], model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # For UDA, we need a domain discriminator and GRL.
    if uda:
        # Create domain discriminator with input dim equal to encoder's latent dimension.
        domain_disc = DomainDiscriminator(
            input_dim=model.encoder.embed_dim,
            hidden_dim=uda_hidden_dim,
            dropout_rate=uda_dropout_rate,
        )
        domain_disc.to(device)
        # Create Gradient Reversal Layer (we can set lambda here, though it might be annealed during training)
        grl = GradientReversal(lambda_=0.0)
        grl.to(device)
        # Use separate optimizers for model and domain discriminator.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )
        optimizer_disc = torch.optim.Adam(
            domain_disc.parameters(),
            lr=uda_disc_lr,
        )
        # Use binary cross entropy loss for domain classification.
        domain_criterion = nn.BCELoss()
        if warmup_epochs > 0:
            total_epochs_uda = training_params["n_epochs"] - warmup_epochs
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_params["factor"],
        patience=training_params["patience"],
        min_lr=training_params["min_lr"],
    )
    warmup_scheduler = WarmupScheduler(
        optimizer, training_params["lr_warmup_epochs"], training_params["lr"]
    )

    if training_params["loss"] == "WeightedRMSELoss":
        criterion = WeightedRMSELoss()
    elif training_params["loss"] == "ZeroInflatedWeightedRMSELoss":
        criterion = ZeroInflatedWeightedRMSELoss(zero_penalty_factor=zero_weight)
    elif training_params["loss"] == "WeightedMSELoss":
        criterion = WeightedMSELoss()
    elif training_params["loss"] == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(
            f"Loss function '{training_params['loss']}' not implemented"
        )

    if verbose:
        model_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if uda:
            disc_p = sum(p.numel() for p in domain_disc.parameters() if p.requires_grad)
            if hasattr(model, "encoder") and hasattr(model, "head"):
                encoder_p = sum(
                    p.numel() for p in model.encoder.parameters() if p.requires_grad
                )
                head_p = sum(
                    p.numel() for p in model.head.parameters() if p.requires_grad
                )
                print(
                    "Number of trainable parameters (encoder + head + domain discriminator):",
                    encoder_p,
                    "+",
                    head_p,
                    "+",
                    disc_p,
                    "=",
                    encoder_p + head_p + disc_p,
                )
            else:
                print(
                    "Number of trainable parameters (model + domain discriminator):",
                    model_p,
                    "+",
                    disc_p,
                    "=",
                    model_p + disc_p,
                )
        else:
            if hasattr(model, "encoder") and hasattr(model, "head"):
                encoder_p = sum(
                    p.numel() for p in model.encoder.parameters() if p.requires_grad
                )
                head_p = sum(
                    p.numel() for p in model.head.parameters() if p.requires_grad
                )
                print(
                    "Number of trainable parameters (encoder + head):",
                    encoder_p,
                    "+",
                    head_p,
                    "=",
                    encoder_p + head_p,
                )
            else:
                print(
                    "Number of trainable parameters:",
                    model_p,
                )

    # Build source data loader
    x_source_train = (
        x_source_train.drop("ID", axis=1)
        if "ID" in x_source_train.columns
        else x_source_train
    )
    y_source_train = (
        y_source_train.drop("ID", axis=1)
        if "ID" in y_source_train.columns
        else y_source_train
    )
    source_dataset = TensorDataset(
        torch.tensor(x_source_train.values, dtype=torch.float32),
        torch.tensor(y_source_train.values, dtype=torch.float32),
    )
    source_loader = DataLoader(
        source_dataset, batch_size=training_params["batch_size"], shuffle=True
    )

    # For UDA: build target loader (unlabeled). We'll create dummy labels.
    if uda:
        target_dataset = TensorDataset(
            torch.tensor(x_target_train.values, dtype=torch.float32),
            torch.zeros(len(x_target_train.values)),
        )  # dummy labels
        target_loader = DataLoader(
            target_dataset, batch_size=training_params["batch_size"], shuffle=True
        )
        if verbose and warmup_epochs > 0:
            print(
                f"Warming up encoder for {warmup_epochs} epochs. No domain adaptation yet."
            )

    # Build validation loaders
    val_loaders = []
    for i, (x_val, y_val) in enumerate(valsets):
        # Create validation loader directly from the provided validation data
        x_val = x_val.drop("ID", axis=1) if "ID" in x_val.columns else x_val
        y_val = y_val.drop("ID", axis=1) if "ID" in y_val.columns else y_val
        val_dataset = TensorDataset(
            torch.tensor(x_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=training_params["batch_size"], shuffle=False
        )
        val_loaders.append(val_loader)
    if labels is None:
        if len(valsets) == 1:
            labels = ["Validation"]
        else:
            labels = [f"Val Set {i}" for i in range(len(val_loaders))]

    if verbose:
        pbar = tqdm(total=training_params["n_epochs"])
    if plot_losses:
        train_losses = []
        val_losses_all = [[] for _ in range(len(val_loaders))]

    criterion_val = WeightedMSELoss()
    epoch_uda = 0
    try:
        for epoch in range(training_params["n_epochs"]):
            if uda:
                if epoch >= warmup_epochs:
                    epoch_uda += 1
                    lambda_ = compute_lambda(epoch_uda / total_epochs_uda, gamma=gamma)
                    grl.lambda_ = lambda_
                else:
                    lambda_ = 0.0
                if verbose and epoch == warmup_epochs:
                    print("Domain adaptation enabled.")
                    # set learning rate of full model to phase learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = uda_phase_lr
                train_loss, disc_loss, reg_loss, disc_acc = train_nn_regressor_uda(
                    model,
                    domain_disc,
                    grl,
                    source_loader,
                    target_loader,
                    optimizer,
                    optimizer_disc,
                    criterion,
                    domain_criterion,
                    device,
                    epoch,
                    warmup_epochs,
                )
                if isinstance(criterion, WeightedMSELoss):
                    reg_loss = reg_loss**0.5
                    train_loss = disc_loss + reg_loss
            else:
                train_loss = train_nn_regressor(
                    model, source_loader, optimizer, criterion, device
                )
                if isinstance(criterion, WeightedMSELoss):
                    train_loss = train_loss**0.5
            # Track validation losses for all validation sets
            val_losses = []
            for i, val_loader in enumerate(val_loaders):
                val_loss = evaluate_nn_regressor(
                    model, val_loader, criterion_val, device
                )
                val_losses.append(val_loss)
                if plot_losses:
                    val_losses_all[i].append(val_loss)

            # Calculate (weighted) mean validation loss (strat_loss * num_samples_in_strat / total_samples for each strat)
            mean_val_loss = np.mean(val_losses) if val_losses else 0

            # Update scheduler with mean validation loss
            if not warmup_scheduler.is_warmup_done():
                warmup_scheduler.step()
            else:
                scheduler.step(mean_val_loss)

            if plot_losses:
                if uda:
                    train_losses.append(reg_loss)
                else:
                    train_losses.append(train_loss)

            if verbose:
                if uda:
                    pbar.set_description(
                        f"Disc Acc: {disc_acc*100:.2f}%, Disc Loss: {disc_loss:.4f}, Lambda: {lambda_:.6f}, Train Reg Loss: {reg_loss:.4f}, Val Loss: {mean_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(
                        f"Train Loss: {train_loss:.4f}, Val Loss: {mean_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                pbar.update(1)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if verbose:
            pbar.close()
            if len(valsets) == 1:
                final_val_rmse = evaluate_nn_regressor(
                    model, val_loaders[0], criterion_val, device
                )
                print(f"Final Validation Weighted RMSE: {final_val_rmse:.4f}")
        if plot_losses:
            plot_loss(train_losses, val_losses_all, labels)
        return model


def submit_nn(
    checkpoint,
    model_class,
    model_params,
    pipeline_params,
    test_path,
    submission_path,
    use_embed=False,
):
    """
    checkpoint: path to the checkpoint file.
    model_class: class of the model to use.
    model_params: parameters for the model.
    pipeline_params: parameters for the pipeline.
    test_path: path to the test data.
    submission_path: path to save the submission file.
    use_embed: whether to use embeddings as features.
    """
    model = instantiate_model(model_class, model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)

    write_submissions(
        model,
        test_path,
        submission_path,
        use_embed=use_embed,
        **pipeline_params,
    )

    return None
