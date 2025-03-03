import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_preprocessing import preprocess_data_2
from feature_engineering import apply_feature_engineering


def write_submissions(
    model,
    test_data_filename,
    output_filename,
    apply_feat_eng=True,
    polynomial_degree=2,
    include_group_interactions=True,
    include_humidity_interactions=True,
    remove_humidity=False,
    feature_selection=False,
    k_features=20,
    use_model_features=True,
    use_embed=True,
    embedder_model=None,
):
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
    if remove_humidity:
        df_test = df_test.drop("Humidity", axis=1)
    df_test_processed = preprocess_data_2(df_test)

    if apply_feat_eng:
        df_test_processed = apply_feature_engineering(
            df_test_processed,
            target_df=None,
            polynomial_degree=polynomial_degree,
            include_group_interactions=include_group_interactions,
            include_humidity_interactions=include_humidity_interactions,
            feature_selection=feature_selection,
            k_features=k_features,
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
    is_selective_linear = hasattr(model, "selected_features") and hasattr(
        model, "feature_names"
    )

    # If it's a SelectiveLinearRegressor and use_model_features is True, ensure we only use the features the model was trained on
    if is_selective_linear and use_model_features:
        # Get all unique features used by the model across all targets
        all_selected_indices = set()
        for indices in model.selected_features:
            all_selected_indices.update(indices)

        # Get the feature names that were used during training
        selected_feature_names = [model.feature_names[i] for i in all_selected_indices]

        # Filter the test dataframe to only include these features plus ID
        available_features = [
            col
            for col in df_test_processed.columns
            if col in selected_feature_names or col == "ID"
        ]
        df_test_processed = df_test_processed[available_features]

        # Update feature columns
        feature_columns = [col for col in df_test_processed.columns if col != "ID"]

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

    # Special handling for SelectiveLinearRegressor
    if hasattr(model, "selected_features") and hasattr(model, "feature_names"):
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            # Create a mapping between model feature indices and test data columns
            feature_mapping = {}
            available_features = set(X.columns)

            # For each target's selected features, map model indices to test data columns
            for target_idx, selected_indices in enumerate(model.selected_features):
                feature_mapping[target_idx] = []
                for model_idx in selected_indices:
                    if model_idx < len(model.feature_names):
                        feature_name = model.feature_names[model_idx]
                        if feature_name in available_features:
                            # Store the column index in the test data
                            test_idx = X.columns.get_loc(feature_name)
                            feature_mapping[target_idx].append((model_idx, test_idx))

            X_array = X.values
        else:
            X_array = X
            # For numpy arrays, we'll use column indices directly
            feature_mapping = {
                i: [(idx, idx) for idx in selected if idx < X_array.shape[1]]
                for i, selected in enumerate(model.selected_features)
            }

        # Initialize predictions array
        predictions = np.zeros((X_array.shape[0], len(model.models)))

        # Generate predictions for each target variable
        for i, submodel in enumerate(model.models):
            mapped_features = feature_mapping.get(i, [])

            if not mapped_features:
                # If no valid features for this target, use zeros as predictions
                predictions[:, i] = np.zeros(X_array.shape[0])
            else:
                # Extract the test data indices for this target
                test_indices = [test_idx for _, test_idx in mapped_features]
                X_selected = X_array[:, test_indices]

                # Make predictions if we have features
                if X_selected.shape[1] > 0:
                    predictions[:, i] = submodel.predict(X_selected)
                else:
                    predictions[:, i] = np.zeros(X_array.shape[0])

        return predictions

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
