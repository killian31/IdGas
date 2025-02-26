import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


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


def full_pipeline(filename_x, filename_y, val_proportion=0.2, reduce_features=True):
    """
    Full pipeline function that:
      - Loads feature and label CSV files.
      - Preprocesses the features.
      - Merges features and labels on 'ID'.
      - Splits the data into training and validation sets.

    Parameters:
      filename_x (str): Path to the features CSV file.
      filename_y (str): Path to the labels CSV file.
      val_proportion (float): Proportion of data to reserve for validation.

    Returns:
      x_train, y_train, x_val, y_val: DataFrames for training and validation.
    """
    df_x = pd.read_csv(filename_x)
    df_y = pd.read_csv(filename_y)

    df_x_processed = preprocess_data(df_x)

    if reduce_features:
        df_x_processed = group_measures(df_x_processed)

    df_merged = pd.merge(df_x_processed, df_y, on="ID")

    target_columns = [col for col in df_y.columns if col != "ID"]
    feature_columns = [col for col in df_x_processed.columns if col != "ID"]

    X = df_merged[["ID"] + feature_columns]
    y = df_merged[target_columns]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=val_proportion, random_state=29
    )

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
        list(zip(x_train.values.astype('float32'), y_train.values.astype('float32'))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        list(zip(x_val.values.astype('float32'), y_val.values.astype('float32'))),
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
