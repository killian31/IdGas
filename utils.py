import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_preprocessing import preprocess_data


def write_submissions(model, test_data_filename, output_filename):
    """
    Generates predictions on test data and writes a CSV submission file.
    Supports both sklearn-like models and PyTorch models.

    Parameters:
      model: Trained model for multi-output regression (sklearn-like or PyTorch).
      test_data_filename (str): Path to the test features CSV file.
      output_filename (str): Path where the submission CSV will be saved.

    The submission CSV will have a header: "ID,c01,c02,...,c23".
    """
    # Load and preprocess the test data
    df_test = pd.read_csv(test_data_filename)
    df_test_processed = preprocess_data(df_test)

    # Extract the ID column for submission
    ids = df_test_processed["ID"].values

    # Check if model is a PyTorch model
    is_torch_model = isinstance(model, torch.nn.Module)

    if is_torch_model:
        # For PyTorch models, use DataLoader for batch processing
        device = next(model.parameters()).device
        model.eval()

        # Create a test DataLoader (batch_size=128 for memory efficiency)
        test_loader = DataLoader(
            df_test_processed.drop(columns=["ID"]).values.astype("float32"),
            batch_size=128,
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
