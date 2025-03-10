import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import (
    preprocess_data_2,
    group_measures,
    splits_to_dataloaders,
    full_pipeline,
)
from feature_engineering import apply_feature_engineering


def write_submissions(
    model,
    test_data_filename,
    output_filename,
    use_embed=False,
    **kwargs,
):
    """
    Generates predictions on test data and writes a CSV submission file.
    Supports both sklearn-like models and PyTorch models.

    Parameters:
      model: Trained model for multi-output regression (sklearn-like or PyTorch).
      test_data_filename (str): Path to the test features CSV file.
      output_filename (str): Path where the submission CSV will be saved.
      kwargs: Additional keyword arguments for data preprocessing.
    Returns:
      None: Writes the submission CSV to the specified path.

    The submission CSV will have a header: "ID,c01,c02,...,c23".
    """
    # Load and preprocess the test data
    df_test_processed = full_pipeline(
        test_data_filename,
        filename_y=None,
        **kwargs,
    )

    if use_embed and (embedder_model is not None):
        embedder_model.eval()
        test_data_tensor = torch.tensor(
            df_test_processed.drop(columns=["ID"]).values.astype("float32"),
            dtype=torch.float32,
        )
        with torch.no_grad():
            test_embeddings = embedder_model.embed(test_data_tensor).cpu().numpy()
        embs_df = pd.DataFrame(
            test_embeddings,
            columns=[f"embed_{i}" for i in range(test_embeddings.shape[1])],
            index=df_test_processed.index,
        )
        df_test_processed = pd.concat([df_test_processed, embs_df], axis=1)
    # Extract feature columns (all columns except ID)
    feature_columns = [col for col in df_test_processed.columns if col != "ID"]
    print(f"Using {len(feature_columns)} features")

    # Extract the ID column for submission
    ids = df_test_processed["ID"].values

    # Check if model is a PyTorch model or SelectiveLinearRegressor
    is_torch_model = isinstance(model, torch.nn.Module)

    if is_torch_model:
        # For PyTorch models, use DataLoader for batch processing
        device = next(model.parameters()).device
        model.eval()

        # Create a test DataLoader (batch_size=128 for memory efficiency)
        test_loader = DataLoader(
            df_test_processed.drop(columns=["ID"]).values.astype("float32"),
            batch_size=512,
            shuffle=False,
        )

        # Generate predictions in batches
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.append(output.cpu().numpy())
        y_pred = np.vstack(predictions)
    else:
        # For sklearn-like models, use the existing predict_model function
        y_pred = predict_model(model, df_test_processed)

    # Create target column names: c01, c02, ..., c23
    target_cols = [f"c{i:02d}" for i in range(1, 24)]

    # Build the submission DataFrame: add ID as the first column
    submission = pd.DataFrame(y_pred, columns=target_cols)
    submission.insert(0, "ID", ids)

    # Write submission to CSV without index
    submission.to_csv(output_filename, index=False)


def predict_model(model, x):
    """
    Generate predictions using the trained model.

    Parameters:
      model: Trained model.
      x (DataFrame): Feature DataFrame (including 'ID', which is dropped if present).

    Returns:
      np.array: Predictions.
    """
    X = x.drop("ID", axis=1) if "ID" in x.columns else x

    if isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
        model.eval()
        # create dataloader
        test_dataset = TensorDataset(
            torch.tensor(X.values, dtype=torch.float32),
            torch.zeros(X.shape[0], dtype=torch.float32),
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        # generate predictions in batches
        predictions = []
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.append(output.cpu().numpy())
        return np.vstack(predictions)

    # For other model types, use standard predict
    return model.predict(X)


def compute_weighted_rmse(y_true, y_pred):
    """
    Compute the weighted RMSE according to the competition metric.

    For each target value:
      - f = 1 if y_true < 0.5
      - f = 1.2 if y_true >= 0.5

    For each sample, the error is the mean over targets of f*(y_pred - y_true)^2.
    The final metric is the square root of the average of these sample errors.

    Parameters:
      y_true (np.array): True target values (shape: [n_samples, n_targets])
      y_pred (np.array): Predicted target values (shape: [n_samples, n_targets])

    Returns:
      float: The weighted RMSE.
    """
    weights = np.where(y_true < 0.5, 1.0, 1.2)
    squared_errors = weights * (y_pred - y_true) ** 2
    sample_errors = np.mean(squared_errors, axis=1)
    metric = np.sqrt(np.mean(sample_errors))
    return metric


def evaluate_model(model, x_val, y_val):
    """
    Evaluate the model on the validation set using the custom weighted RMSE.

    Parameters:
      model: Trained model.
      x_val (DataFrame): Validation features.
      y_val (DataFrame): Validation targets.

    Returns:
      float: Weighted RMSE on the validation set.
    """
    y_pred = predict_model(model, x_val)
    metric = compute_weighted_rmse(y_val.values, y_pred)
    return metric


def stratify_by_humidity(x_val, y_val, n_strata=5):
    """
    Stratify validation data into n subsets based on humidity values.
    Returns lists of x and y dataframes for each stratum.
    """
    # Calculate percentile boundaries to ensure no empty sets
    percentiles = np.linspace(0, 100, n_strata + 1)
    boundaries = np.percentile(x_val["Humidity"], percentiles)

    x_strata = []
    y_strata = []
    labels = []  # for plotting

    # Create strata using the boundaries
    for i in range(n_strata):
        if i == n_strata - 1:
            # Include the right boundary for the last stratum
            mask = (x_val["Humidity"] >= boundaries[i]) & (
                x_val["Humidity"] <= boundaries[i + 1]
            )
        else:
            mask = (x_val["Humidity"] >= boundaries[i]) & (
                x_val["Humidity"] < boundaries[i + 1]
            )

        x_strata.append(x_val[mask])
        y_strata.append(y_val[mask])
        labels.append(f"H in [{boundaries[i]:.2f}, {boundaries[i+1]:.2f}]")

    return x_strata, y_strata, labels


def create_humidity_subsets(x_val, y_val):
    """Create validation subsets based on humidity levels.

    Parameters:
        x_val (DataFrame): Validation features
        y_val (DataFrame): Validation targets

    Returns:
        list: List of tuples (x_subset, y_subset) for each humidity range
    """
    humidity_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    subsets = []

    for start, end in humidity_ranges:
        # Get indices where humidity falls within the current range
        mask = (
            (x_val["Humidity_x"] >= start) & (x_val["Humidity_x"] < end)
            if "Humidity_x" in x_val.columns
            else (x_val["Humidity"] >= start) & (x_val["Humidity"] < end)
        )
        x_subset = x_val[mask]
        y_subset = y_val.loc[x_subset.index]
        subsets.append((x_subset, y_subset))

    return subsets
