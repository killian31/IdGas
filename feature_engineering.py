import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA


def create_power_features(df, degree=2):
    """
    Creates power features from the input dataframe.

    Parameters:
        df (DataFrame): Input dataframe with features
        degree (int): Power of features
    Returns:
        DataFrame: Original dataframe with polynomial features added
    """
    if degree < 2:
        return df
    transformed_df = df.copy()
    for col in df.columns:
        if col == "ID":
            continue
        for d in range(2, degree + 1):
            transformed_df[f"{col}_power_{d}"] = df[col] ** d
    for d in range(2, degree + 1):
        transformed_df[f"Humidity_root_{d}"] = np.abs(df["Humidity"]) ** (1 / d)
    return transformed_df


def create_sensor_group_interactions(df):
    """
    Creates interaction features between sensor groups.

    Parameters:
        df (DataFrame): Input dataframe with features

    Returns:
        DataFrame: Original dataframe with interaction features added
    """
    df_result = df.copy()

    # Define sensor groups
    m12_15_group = ["M12", "M13", "M14", "M15"]
    m4_7_group = ["M4", "M5", "M6", "M7"]
    rs_group = ["R", "S1", "S2", "S3"]

    # Create group aggregations
    for group, prefix in [
        (m12_15_group, "M12_15"),
        (m4_7_group, "M4_7"),
        (rs_group, "RS"),
    ]:
        if all(col in df.columns for col in group):
            df_result[f"{prefix}_mean"] = df[group].mean(axis=1)
            df_result[f"{prefix}_std"] = df[group].std(axis=1)
            df_result[f"{prefix}_max"] = df[group].max(axis=1)
            df_result[f"{prefix}_min"] = df[group].min(axis=1)
            df_result[f"{prefix}_range"] = (
                df_result[f"{prefix}_max"] - df_result[f"{prefix}_min"]
            )

    # Create cross-group interactions
    if "M12_15_mean" in df_result.columns and "M4_7_mean" in df_result.columns:
        df_result["M12_15_M4_7_ratio"] = df_result["M12_15_mean"] / (
            df_result["M4_7_mean"] + 1e-8
        )
        df_result["M12_15_M4_7_product"] = (
            df_result["M12_15_mean"] * df_result["M4_7_mean"]
        )

    if "M12_15_mean" in df_result.columns and "RS_mean" in df_result.columns:
        df_result["M12_15_RS_ratio"] = df_result["M12_15_mean"] / (
            df_result["RS_mean"] + 1e-8
        )
        df_result["M12_15_RS_product"] = df_result["M12_15_mean"] * df_result["RS_mean"]

    if "M4_7_mean" in df_result.columns and "RS_mean" in df_result.columns:
        df_result["M4_7_RS_ratio"] = df_result["M4_7_mean"] / (
            df_result["RS_mean"] + 1e-8
        )
        df_result["M4_7_RS_product"] = df_result["M4_7_mean"] * df_result["RS_mean"]

    return df_result


def create_humidity_interactions(df):
    """
    Creates interaction features between humidity and sensor readings.

    Parameters:
        df (DataFrame): Input dataframe with features

    Returns:
        DataFrame: Original dataframe with humidity interaction features added
    """
    df_result = df.copy()

    if "Humidity" not in df.columns:
        return df_result

    # Create humidity transformations if not already present
    if "Humidity_power_2" not in df.columns:
        df_result["Humidity_power_2"] = df_result["Humidity"] ** 2
    if "Humidity_root_2" not in df.columns:
        df_result["Humidity_root_2"] = np.sqrt(np.abs(df_result["Humidity"]))

    # Create interactions with sensor groups
    sensor_cols = [col for col in df.columns if col not in ["ID", "Humidity"]]

    for col in sensor_cols:
        df_result[f"Humidity_{col}_product"] = df_result["Humidity"] * df_result[col]

    for col in sensor_cols:
        df_result[f"Humidity_{col}_ratio"] = df_result[col] / (
            df_result["Humidity"] + 1e-8
        )

    # Create interactions with group means if they exist
    for group_mean in ["M12_15_mean", "M4_7_mean", "RS_mean"]:
        if group_mean in df_result.columns:
            df_result[f"Humidity_{group_mean}_product"] = (
                df_result["Humidity"] * df_result[group_mean]
            )
            df_result[f"Humidity_{group_mean}_ratio"] = df_result[group_mean] / (
                df_result["Humidity"] + 1e-8
            )

    return df_result


def select_features(X, y, method="mutual_info", k=20):
    """
    Selects the top k features based on the specified method.

    Parameters:
        X (DataFrame): Feature dataframe
        y (DataFrame/Series): Target variable
        method (str): Feature selection method ('mutual_info' or 'f_regression')
        k (int): Number of features to select

    Returns:
        DataFrame: Dataframe with selected features
    """
    # Preserve ID column if it exists
    id_col = None
    if "ID" in X.columns:
        id_col = X["ID"].copy()
        X = X.drop(columns=["ID"])

    # Select features
    if method == "mutual_info":
        selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    else:  # f_regression
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))

    # If y has multiple columns (multi-target), use the mean for feature selection
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        y_mean = y.mean(axis=1)
        X_selected = selector.fit_transform(X, y_mean)
    else:
        X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    X_result = pd.DataFrame(X_selected, columns=selected_features)

    # Add back ID column
    if id_col is not None:
        X_result["ID"] = id_col

    return X_result


def apply_feature_engineering(
    df,
    target_df=None,
    power_degree=2,
    include_group_interactions=True,
    include_humidity_interactions=True,
    feature_selection=True,
    k_features=20,
):
    """
    Applies comprehensive feature engineering to the input dataframe.

    Parameters:
        df (DataFrame): Input dataframe with features
        target_df (DataFrame, optional): Target dataframe for feature selection
        power_degree (int): Degree for power features
        include_group_interactions (bool): Whether to include sensor group interactions
        include_humidity_interactions (bool): Whether to include humidity interactions
        feature_selection (bool): Whether to perform feature selection
        k_features (int): Number of features to select if feature_selection is True

    Returns:
        DataFrame: Dataframe with engineered features
    """
    result_df = df.copy()
    # Step 1: Create sensor group interactions
    if include_group_interactions:
        result_df = create_sensor_group_interactions(result_df)

    # Step 2: Create humidity interactions
    if include_humidity_interactions and "Humidity" in result_df.columns:
        result_df = create_humidity_interactions(result_df)

    # Step 3: Create power features (only for non-aggregated features to avoid explosion)
    if power_degree > 1:
        sensor_cols = [col for col in df.columns if col != "ID"]
        if sensor_cols:
            poly_df = create_power_features(
                df[sensor_cols],
                degree=power_degree,
            )
            # Add ID back if they exist
            if "ID" in df.columns:
                poly_df["ID"] = df["ID"]

            # Merge with result_df, keeping only new polynomial features
            original_cols = result_df.columns
            result_df = poly_df
            # pd.merge(
            #    result_df, poly_df, on=["ID"] if "ID" in df.columns else None
            # )
            # Remove duplicate columns that might have been added
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # Step 4: Feature selection if target is provided
    if feature_selection and target_df is not None:
        result_df = select_features(
            result_df, target_df, method="mutual_info", k=k_features
        )

    return result_df
