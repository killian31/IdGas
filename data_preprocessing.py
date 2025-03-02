import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader
from feature_engineering import apply_feature_engineering


def preprocess_data(df):
    """
    Preprocesses the dataset by applying the appropriate transformations:
    - StandardScaler to M12-M15 (Gaussian distribution)
    - Log transformation + StandardScaler to M4-M7, R, S1-S3 (preserving structure without distorting negatives)
    - Leaves Humidity and ID unchanged
    """
    df_transformed = df.copy()

    standard_scaler_features = ["M12", "M13", "M14", "M15"]
    log_standard_scaler_features = [
        "M4",
        "M5",
        "M6",
        "M7",
        "R",
        "S1",
        "S2",
        "S3",
    ]

    scaler = StandardScaler()
    df_transformed[standard_scaler_features] = scaler.fit_transform(
        df_transformed[standard_scaler_features]
    )

    df_transformed[log_standard_scaler_features] = np.sign(
        df_transformed[log_standard_scaler_features]
    ) * np.log1p(np.abs(df_transformed[log_standard_scaler_features]))

    log_scaler = StandardScaler()
    df_transformed[log_standard_scaler_features] = log_scaler.fit_transform(
        df_transformed[log_standard_scaler_features]
    )

    return df_transformed


def preprocess_data_2(df, cap_quantiles=(0.01, 0.99)):
    df = df.copy()
    num_cols = [c for c in df.columns if c not in ["ID", "Humidity"]]
    for c in num_cols:
        low = df[c].quantile(cap_quantiles[0])
        high = df[c].quantile(cap_quantiles[1])
        df[c] = np.clip(df[c], low, high)
    m12_15 = ["M12", "M13", "M14", "M15"]
    rs = RobustScaler()
    df[m12_15] = rs.fit_transform(df[m12_15])
    log_cols = ["M4", "M5", "M6", "M7", "R", "S1", "S2", "S3"]
    df[log_cols] = np.sign(df[log_cols]) * np.log1p(np.abs(df[log_cols]))
    ss = StandardScaler()
    df[log_cols] = ss.fit_transform(df[log_cols])
    return df


def augment_data(
    df,
    humidity_col="Humidity",
    sensor_cols=None,
    target_humidity_range=(0.0, 1.2),
    n_copies=1,
    noise_std=0.05,
):
    """
    Creates augmented samples by:
      1) Randomly scaling humidity into a target range.
      2) Adjusting sensor values accordingly with small random noise.

    df: input DataFrame
    humidity_col: name of the humidity column
    sensor_cols: list of sensor columns to modify
    target_humidity_range: range of humidity to sample from
    n_copies: how many augmented copies per original row
    noise_std: std dev of Gaussian noise added to sensor columns

    Returns: original df appended with new augmented rows
    """
    if sensor_cols is None:
        # Exclude ID and label columns if present
        sensor_cols = [c for c in df.columns if c not in [humidity_col, "ID"]]

    augmented_rows = []
    for _ in range(n_copies):
        # Copy the original data
        df_aug = df.copy()

        # Sample new humidity scalars
        original_hum = df_aug[humidity_col].values
        # e.g., uniform sampling within a broader range:
        new_hum = np.random.uniform(*target_humidity_range, size=len(df_aug))

        # Compute ratio
        ratio = (new_hum + 1e-8) / (original_hum + 1e-8)

        # Scale sensor values by ratio^(some factor) + noise
        for c in sensor_cols:
            df_aug[c] = df_aug[c].values * ratio**0.5  # example mild effect
            df_aug[c] += np.random.normal(0, noise_std, size=len(df_aug))

        df_aug[humidity_col] = new_hum
        augmented_rows.append(df_aug)

    # Concatenate augmented data
    df_out = pd.concat([df] + augmented_rows, ignore_index=True)
    return df_out


def group_measures(df, verbose=True):
    """
    Groups M12 to M15 into a single measure, M4 to M7 into 2 variables using PCA.
    """
    df_grouped = df.copy()

    pca_12_to_15 = PCA(n_components=1)
    df_grouped["M12-M15"] = pca_12_to_15.fit_transform(
        df_grouped[["M12", "M13", "M14", "M15"]]
    )
    if verbose:
        print(
            "Explained variance for M12-M15:",
            np.cumsum(pca_12_to_15.explained_variance_ratio_)[-1],
        )
    pca_4_to_7 = PCA(n_components=2)
    df_grouped[["M4-M7_1", "M4-M7_2"]] = pca_4_to_7.fit_transform(
        df_grouped[["M4", "M5", "M6", "M7"]]
    )
    if verbose:
        print(
            "Explained variance for M4-M7 (cumulative):",
            np.cumsum(pca_4_to_7.explained_variance_ratio_)[-1],
        )

    df_grouped.drop(
        columns=["M12", "M13", "M14", "M15", "M4", "M5", "M6", "M7"], inplace=True
    )

    return df_grouped


def full_pipeline(
    filename_x, filename_y, val_proportion=0.2, reduce_features=False, augment=False,
    apply_feat_eng=True, polynomial_degree=2, include_group_interactions=True,
    include_humidity_interactions=True, feature_selection=True, k_features=20
):
    """
    Full pipeline function that:
      - Loads feature and label CSV files.
      - Preprocesses the features.
      - Applies feature engineering (optional).
      - Merges features and labels on 'ID'.
      - Splits the data into training and validation sets.
      - Applies data augmentation only to the training set if requested.

    Parameters:
      filename_x (str): Path to the features CSV file.
      filename_y (str): Path to the labels CSV file.
      val_proportion (float): Proportion of data to reserve for validation.
      reduce_features (bool): If True, reduces the number of features using PCA.
      augment (bool): If True, augments data with synthetic samples (only applied to training data).
      apply_feat_eng (bool): If True, applies feature engineering techniques.
      polynomial_degree (int): Degree for polynomial features.
      include_group_interactions (bool): Whether to include sensor group interactions.
      include_humidity_interactions (bool): Whether to include humidity interactions.
      feature_selection (bool): Whether to perform feature selection.
      k_features (int): Number of features to select if feature_selection is True.

    Returns:
      x_train, y_train, x_val, y_val: DataFrames for training and validation.
    """
    df_x = pd.read_csv(filename_x)
    df_y = pd.read_csv(filename_y)

    df_x_processed = preprocess_data_2(df_x)

    if reduce_features:
        df_x_processed = group_measures(df_x_processed)

     # Apply feature engineering if requested
    if apply_feat_eng:
        df_x_processed = apply_feature_engineering(
            df_x_processed,
            target_df=df_y,
            polynomial_degree=polynomial_degree,
            include_group_interactions=include_group_interactions,
            include_humidity_interactions=include_humidity_interactions,
            feature_selection=feature_selection,
            k_features=k_features
        )
        
    df_merged = pd.merge(df_x_processed, df_y, on="ID")

    target_columns = [col for col in df_y.columns if col != "ID"]
    feature_columns = [col for col in df_x_processed.columns if col != "ID"]

    X = df_merged[["ID"] + feature_columns]
    y = df_merged[target_columns]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=val_proportion, random_state=29
    )
    
    # Apply data augmentation only to the training set if requested
    if augment:
        # Extract features without ID for augmentation
        train_features = x_train.drop(columns=["ID"]) if "ID" in x_train.columns else x_train.copy()
        
        # Augment only the training features
        augmented_features = augment_data(
            train_features,
            humidity_col="Humidity",
            sensor_cols=[
                "M4",
                "M5",
                "M6",
                "M7",
                "R",
                "S1",
                "S2",
                "S3",
                "M12",
                "M13",
                "M14",
                "M15",
            ],
            target_humidity_range=(0.0, 1.2),
            n_copies=1,
            noise_std=0.02,
        )
        
        # Generate new IDs for augmented data if needed
        if "ID" in x_train.columns:
            max_id = x_train["ID"].max()
            new_ids = pd.DataFrame({"ID": range(max_id + 1, max_id + 1 + len(augmented_features) - len(train_features))})
            augmented_features = pd.concat([new_ids, augmented_features], axis=1)
        
        # Create corresponding labels for augmented data (copy original labels)
        augmented_labels = pd.concat([y_train] * 2, ignore_index=True)[len(y_train):]
        
        # Combine original and augmented data
        x_train = pd.concat([x_train, augmented_features], ignore_index=True)
        y_train = pd.concat([y_train, augmented_labels], ignore_index=True)

    return x_train, y_train, x_val, y_val


def splits_to_dataloaders(x_train, y_train, x_val, y_val, batch_size=32, num_workers=0):
    """
    Converts training and validation DataFrames to PyTorch DataLoaders.
    Parameters:
      x_train, y_train, x_val, y_val (DataFrame): Training and validation data.
      batch_size (int): Size of each batch.
    Returns:
      train_loader, val_loader (DataLoader): PyTorch DataLoaders for training and validation.
    """
    if "ID" in x_train.columns:
        x_train = x_train.drop(columns=["ID"])
    if "ID" in x_val.columns:
        x_val = x_val.drop(columns=["ID"])
    if "ID" in y_train.columns:
        y_train = y_train.drop(columns=["ID"])
    if "ID" in y_val.columns:
        y_val = y_val.drop(columns=["ID"])
    train_loader = DataLoader(
        list(zip(x_train.values.astype("float32"), y_train.values.astype("float32"))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        list(zip(x_val.values.astype("float32"), y_val.values.astype("float32"))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = full_pipeline(
        "data/x_train_T9QMMVq.csv",
        "data/y_train_R0MqWmu.csv",
        val_proportion=0.5,
        reduce_features=False,
    )

    train_loader, val_loader = splits_to_dataloaders(
        x_train, y_train, x_val, y_val, batch_size=256
    )
    for data, label in val_loader:
        print(data.shape, label.shape)
        print(data[:, 0].min(), data[:, 0].max())
        break
