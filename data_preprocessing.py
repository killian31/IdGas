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
    - Log to Humidity
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
    df_transformed["Humidity"] = np.log1p(df_transformed["Humidity"])
    return df_transformed


def preprocess_data_2(df, cap_quantiles=(0.01, 0.99)):
    df = df.copy()
    standard_cols = ["M4", "M5", "M6", "M7", "M12", "M13", "M14", "M15"]
    log_cols = ["R", "S1", "S2", "S3"]
    ss = RobustScaler()
    df[standard_cols] = ss.fit_transform(df[standard_cols])
    df[log_cols] = np.log(df[log_cols] + 1e-4)
    # ss = StandardScaler()
    # df[log_cols] = ss.fit_transform(df[log_cols])

    df["Humidity"] = np.log(df["Humidity"] + 1e-4)

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


def remove_outliers_iqr(df, columns, k=1.5):
    """
    Remove outliers from specified columns using the IQR method.
    k: multiplier for IQR range (default 1.5 for mild outliers)
    """
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
    return df_clean


def split_classif_reg_targets(y):
    """
    Splits the multi-dimensional regression targets into classification and regression targets.

    For each row in the input DataFrame y (shape: [n_samples, n_targets]):
      - The classification target (y_classif) is a binary vector (same shape as y) where each entry is 1 if
        the corresponding value in y is > 0.0, and 0 otherwise.
      - The regression target (y_reg) is a scalar (stored as a one-column DataFrame) representing the common
        non-zero value among the active columns, or 0.0 if no column is active.

    Parameters:
      y (pd.DataFrame): DataFrame with shape (n_samples, n_targets) where each row is a 23-dim vector of floats.
                        Active columns (value > 0.0) are assumed to all have the same non-zero value.

    Returns:
      tuple: (y_classif, y_reg)
        - y_classif (pd.DataFrame): Binary classification targets (same shape as y).
        - y_reg (pd.DataFrame): Regression targets (shape: [n_samples, 1]) with column name "reg_target".
    """
    # Create binary targets: 1 if value > 0, otherwise 0
    y_classif = (y > 0.0).astype(int)

    # For regression target: assume all active values are the same in a given row.
    # If no value is active (i.e., all zeros), return 0.0.
    def get_reg_val(row):
        active_vals = row[row > 0.0]
        if len(active_vals) > 0:
            return active_vals.iloc[0]
        return 0.0

    # Apply function row-wise and convert to a DataFrame with a single column.
    y_reg = y.apply(get_reg_val, axis=1).to_frame(name="reg_target")

    return y_classif, y_reg


def full_pipeline(
    filename_x,
    filename_y=None,
    split_humidity=False,
    h_threshold=0.5,
    split=False,
    val_proportion=0.5,
    rescale=False,
    reduce_features=False,
    augment=False,
    apply_feat_eng=False,
    polynomial_degree=2,
    include_group_interactions=True,
    include_humidity_interactions=True,
    feature_selection=False,
    k_features=20,
    remove_outliers=False,
    k=3.0,
    binarize_humidity=False,
    n_bins=10,
    remove_humidity=False,
):
    """
    Full pipeline function that:
      - Loads feature and label CSV files.
      - Removes Humidity column if requested.
      - Removes outliers if requested (using IQR method).
      - Preprocesses the features.
      - Applies feature engineering (optional).
      - Merges features and labels on 'ID' (if labels provided).
      - Splits the data into training and validation sets (if labels provided).
      - Applies data augmentation only to the training set if requested (if labels provided).

    Parameters:
      filename_x (str): Path to the features CSV file.
      filename_y (str, optional): Path to the labels CSV file. If None, processes data for inference.
      split (bool): If True and filename_y provided, splits data into training and validation sets.
      val_proportion (float): Proportion of data to reserve for validation.
      reduce_features (bool): If True, reduces the number of features using PCA.
      augment (bool): If True and filename_y provided, augments data with synthetic samples.
      apply_feat_eng (bool): If True, applies feature engineering techniques.
      polynomial_degree (int): Degree for polynomial features.
      include_group_interactions (bool): Whether to include sensor group interactions.
      include_humidity_interactions (bool): Whether to include humidity interactions.
      feature_selection (bool): Whether to perform feature selection (only if filename_y provided).
      k_features (int): Number of features to select if feature_selection is True.
      remove_outliers (bool): If True, removes outliers using IQR method before preprocessing.
      k (float): Multiplier for IQR range in outlier removal.
      binarize_humidity (bool): If True, binarizes the Humidity column.
      n_bins (int): Number of bins for binarization.
      remove_humidity (bool): If True, removes the Humidity column from the dataset.

    Returns:
      If filename_y is provided:
        x_train, y_train, x_val, y_val: DataFrames for training and validation.
      If filename_y is None:
        x_processed: DataFrame with processed features for inference.
    """
    df_x = pd.read_csv(filename_x)
    df_y = pd.read_csv(filename_y) if filename_y is not None else None

    if remove_humidity:
        df_x = df_x.drop("Humidity", axis=1)

    if binarize_humidity and "Humidity" in df_x.columns:
        df_x["Humidity"] = pd.cut(df_x["Humidity"], bins=n_bins, labels=False)

    # Remove outliers if requested
    if remove_outliers:
        sensor_cols = [
            "M4",
            "M5",
            "M6",
            "M7",
            "M12",
            "M13",
            "M14",
            "M15",
            "R",
            "S1",
            "S2",
            "S3",
        ]
        df_x = remove_outliers_iqr(df_x, sensor_cols, k=k)
        # Keep only corresponding rows in df_y if available
        if df_y is not None:
            df_y = df_y[df_y["ID"].isin(df_x["ID"])]

    if rescale:
        df_x_processed = preprocess_data(df_x)
    else:
        df_x_processed = df_x

    if reduce_features:
        df_x_processed = group_measures(df_x_processed)

    # Apply feature engineering if requested
    if apply_feat_eng:
        df_x_processed = apply_feature_engineering(
            df_x_processed,
            target_df=df_y,  # Pass None for inference
            polynomial_degree=polynomial_degree,
            include_group_interactions=include_group_interactions,
            include_humidity_interactions=include_humidity_interactions,
            feature_selection=feature_selection
            and df_y is not None,  # Only do feature selection if we have labels
            k_features=k_features,
        )

    # If no labels provided, return processed features
    if df_y is None:
        return df_x_processed

    # From here on, we know we have labels
    df_merged = pd.merge(df_x_processed, df_y, on="ID")

    target_columns = [col for col in df_y.columns if col != "ID"]
    feature_columns = [col for col in df_x_processed.columns if col != "ID"]

    X = df_merged[["ID"] + feature_columns]
    y = df_merged[target_columns]

    assert not (
        split and split_humidity
    ), "Cannot split and split_humidity at the same time"

    if split:
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=val_proportion, random_state=29
        )
    else:
        x_train, y_train, x_val, y_val = X, y, None, None

    if split_humidity:
        # split the training set by humidity using h_humidity (below is kept for training)
        # then add split_proportion of the high humidity to the validation set, the rest to the training set
        x_train_low_h = (
            x_train[x_train["Humidity_x"] < h_threshold]
            if "Humidity_x" in x_train.columns
            else x_train[x_train["Humidity"] < h_threshold]
        )
        x_train_high_h = (
            x_train[x_train["Humidity_x"] >= h_threshold]
            if "Humidity_x" in x_train.columns
            else x_train[x_train["Humidity"] >= h_threshold]
        )
        y_train_low_h = (
            y_train[x_train["Humidity_x"] < h_threshold]
            if "Humidity_x" in x_train.columns
            else y_train[x_train["Humidity"] < h_threshold]
        )
        y_train_high_h = (
            y_train[x_train["Humidity_x"] >= h_threshold]
            if "Humidity_x" in x_train.columns
            else y_train[x_train["Humidity"] >= h_threshold]
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_high_h, y_train_high_h, test_size=val_proportion, random_state=29
        )
        x_train = pd.concat([x_train_low_h, x_train])
        y_train = pd.concat([y_train_low_h, y_train])

    # Apply data augmentation only to the training set if requested
    if augment:
        # Extract features without ID for augmentation
        train_features = (
            x_train.drop(columns=["ID"]) if "ID" in x_train.columns else x_train.copy()
        )

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
            new_ids = pd.DataFrame(
                {
                    "ID": range(
                        max_id + 1,
                        max_id + 1 + len(augmented_features) - len(train_features),
                    )
                }
            )
            augmented_features = pd.concat([new_ids, augmented_features], axis=1)

        # Create corresponding labels for augmented data (copy original labels)
        augmented_labels = pd.concat([y_train] * 2, ignore_index=True)[len(y_train) :]

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


def make_groups(x_train, y_train, x_val, y_val, x_test, plot=True):
    th = 0.0
    var = "M7"

    groups = [
        (0.0, 0.053),
        (0.053, 0.211),
        (0.211, 0.51),
        (0.51, 0.672),
        (0.672, 0.8841),
        (0.8841, 1.0),
    ]
    # apply log1p to groups
    groups = [(np.log1p(g[0]), np.log1p(g[1])) for g in groups]

    if plot:
        plt.figure(figsize=(20, 5))
        x_train_plot = x_train[x_train["Humidity"] > th]
        x_test_plot = x_test[x_test["Humidity"] > th]
        plt.scatter(
            x_train_plot["Humidity"], x_train_plot[var], color="blue", alpha=0.1, s=10
        )
        plt.scatter(
            x_test_plot["Humidity"], x_test_plot[var], color="red", alpha=0.1, s=10
        )
        # plot vertical lines for each group
        xs = np.linspace(th, 1.0, 30)
        for i, g in enumerate(groups):
            if i != 0:
                plt.axvline(x=g[0], color="black", linestyle="--")
        plt.grid()
        # add more details to x axis scale
        plt.xticks(xs)
        plt.show()

    training_sets = []
    validation_sets = []
    for i, g in enumerate(groups):
        training_sets.append(
            (
                x_train[(x_train["Humidity"] > g[0]) & (x_train["Humidity"] <= g[1])],
                y_train[(x_train["Humidity"] > g[0]) & (x_train["Humidity"] <= g[1])],
            )
        )
        validation_sets.append(
            (
                x_val[(x_val["Humidity"] > g[0]) & (x_val["Humidity"] <= g[1])],
                y_val[(x_val["Humidity"] > g[0]) & (x_val["Humidity"] <= g[1])],
            )
        )
    for i, g in enumerate(groups):
        print(f"Group {i+1}: {g}")
        print(f"Training set: {training_sets[i][0].shape}")
        print(f"Validation set: {validation_sets[i][0].shape}")
        print()

    return training_sets, validation_sets
