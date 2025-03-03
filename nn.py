import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_preprocessing import splits_to_dataloaders
from residual_attention_net import RAMTNet


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

        # Fuse
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


class WeightedRMSELoss(nn.Module):
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, output, target):
        weights = torch.where(target < 0.5, 1.0, 1.2)
        squared_errors = weights * (output - target) ** 2
        sample_errors = torch.mean(squared_errors, dim=1)
        loss = torch.sqrt(torch.mean(sample_errors))
        return loss


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    return val_loss / len(val_loader)


def run_experiment(
    x_train, y_train, x_val, y_val, params=None, verbose=False, plot_losses=False
):
    if params is None:
        params = {
            "model_class": "GasDetectionModel",
            "model_params": {
                "hidden_dim": 256,
                "num_output": 23,
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
                "loss": "WeightedRMSELoss",
            },
        }

    model_params = params["model_params"]
    training_params = params["training_params"]
    if params["model_class"] == "NNRegressor":
        model = NNRegressor(**model_params)
    elif params["model_class"] == "GasDetectionModel":
        model = GasDetectionModel(**model_params)
    elif params["model_class"] == "NNRegressorV2":
        model = NNRegressorV2(**model_params)
    elif params["model_class"] == "BasicDeepRegressor":
        model = BasicDeepRegressor(**model_params)
    elif params["model_class"] == "RAMTNet":
        model = RAMTNet(**model_params)
    else:
        raise NotImplementedError(
            f"Model class '{params["model_class"]}' not implemented"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
    if training_params["loss"] == "WeightedRMSELoss":
        criterion = WeightedRMSELoss()
    elif training_params["loss"] == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(
            f"Loss function '{training_params['loss']}' not implemented"
        )
    if verbose:
        print(
            "Number of trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    train_loader, val_loader = splits_to_dataloaders(
        x_train, y_train, x_val, y_val, training_params["batch_size"]
    )

    if verbose:
        pbar = tqdm(total=training_params["n_epochs"])
    if plot_losses:
        train_losses = []
        val_losses = []
    criterion_val = WeightedRMSELoss()
    for epoch in range(training_params["n_epochs"]):
        train_loss = train_nn_regressor(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = evaluate_nn_regressor(model, val_loader, criterion_val, device)
        scheduler.step(val_loss)
        if plot_losses:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        if verbose:
            pbar.set_description(
                f"Epoch {epoch+1}/{training_params['n_epochs']} - Train Loss: {train_loss:.4f}, Val Loss (weighted): {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    val_loss = evaluate_nn_regressor(model, val_loader, criterion_val, device)
    print("Validation Weighted RMSE: {:.4f}".format(val_loss))
    if plot_losses:
        plot_loss(train_losses, val_losses)
    return model
