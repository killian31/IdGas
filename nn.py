import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data_preprocessing import splits_to_dataloaders


class NNRegressor(nn.Module):
    def __init__(self, hidden_dim=128, num_output=23, bias=True, dropout_rate=0.3):
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
    params = {
        "model_params": {
            "hidden_dim": 256,
            "num_output": 23,
            "bias": True,
            "dropout_rate": 0.3,
        },
        "training_param": {
            "n_epochs": 100,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 128,
        },
    }

    model_params = params["model_params"]
    training_param = params["training_param"]
    model = NNRegressor(**model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_param["lr"],
        weight_decay=training_param["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=verbose
    )
    criterion = WeightedRMSELoss()
    if verbose:
        print(
            "Number of trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    train_loader, val_loader = splits_to_dataloaders(
        x_train, y_train, x_val, y_val, training_param["batch_size"]
    )

    if verbose:
        pbar = tqdm(total=training_param["n_epochs"])
    if plot_losses:
        train_losses = []
        val_losses = []
    for epoch in range(training_param["n_epochs"]):
        train_loss = train_nn_regressor(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = evaluate_nn_regressor(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if plot_losses:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        if verbose:
            pbar.set_description(
                f"Epoch {epoch+1}/{training_param['n_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    val_loss = evaluate_nn_regressor(model, val_loader, criterion, device)
    print("Validation Weighted RMSE: {:.4f}".format(val_loss))
    if plot_losses:
        plot_loss(train_losses, val_losses)
    return model
