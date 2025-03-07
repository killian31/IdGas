import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


def load_csv_data(file_path):
    return pd.read_csv(file_path)


def basic_info(df):
    print("DataFrame Info:")
    df.info()
    print("\nDataFrame Head:")
    print(df.head())


def descriptive_stats(df):
    return df.describe()


def plot_missing_values(df, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    msno.matrix(df)
    plt.title("Missing Values Matrix")
    plt.show()


def plot_feature_distributions(df, bins=50, exclude=["ID"], figsize=(12, 12)):
    """
    Plot the distribution of each feature in the DataFrame, on a grid of subplots.
    """
    features = df.columns
    n_features = len(features)
    n_cols = int(np.ceil(np.sqrt(n_features)))

    fig, axes = plt.subplots(n_cols, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        if feature in exclude:
            continue

        ax = axes[i]
        sns.histplot(df[feature], bins=bins, ax=ax, kde=True)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def boxplots(df, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=df)
    plt.title("Boxplots of Features")
    plt.xticks(rotation=45)
    plt.show()


def plot_correlations(df, figsize=(12, 10), annot=False):
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    return corr


def plot_pairplot(df, features, hue=None):
    if features is None:
        sns.pairplot(df)
    else:
        sns.pairplot(df[features], hue=hue)
    plt.suptitle("Pairplot", y=1.02)
    plt.show()


def plot_feature_target_correlations(features_df, target_df, figsize=(12, 10)):
    if "ID" in features_df.columns and "ID" in target_df.columns:
        merged_df = pd.merge(features_df, target_df, on="ID")
    else:
        merged_df = pd.concat([features_df, target_df], axis=1)

    features = features_df.columns.difference(["ID"])
    targets = target_df.columns.difference(["ID"])
    corr_df = pd.DataFrame(index=features, columns=targets)

    for f in features:
        for t in targets:
            corr_df.loc[f, t] = merged_df[f].corr(merged_df[t])

    corr_df = corr_df.astype(float)

    plt.figure(figsize=figsize)
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Feature vs Target Correlations")
    plt.show()

    return corr_df


def find_value_intervals(df, proportion=0.99, print_results=True):
    """
    Returns, for each numeric column in df, the [lower, upper] interval
    that captures the specified proportion (default 99%) of the data.
    For example, proportion=0.99 returns the central 99% interval.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe.
    proportion : float, optional
        Proportion of data to capture in the interval (between 0 and 1).

    Returns:
    --------
    intervals : dict
        A dictionary where each key is a column name and the value is a tuple
        (lower_bound, upper_bound).
    """
    intervals = {}
    # Example: proportion=0.99 -> lower_q=0.005, upper_q=0.995
    lower_q = (1 - proportion) / 2
    upper_q = 1 - lower_q

    for col in df.select_dtypes(include=["float", "int"]).columns:
        lower_bound = df[col].quantile(lower_q)
        upper_bound = df[col].quantile(upper_q)
        intervals[col] = (lower_bound, upper_bound)
        if print_results:
            print(
                f"{proportion*100:.0f}% of the {col} values are in the interval [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
            print(f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")

    return intervals
