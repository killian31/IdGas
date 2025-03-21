from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import itertools
from utils import evaluate_model, create_humidity_subsets


def train_random_forest(x_train, y_train, params=None):
    """
    Train a RandomForestRegressor for multi-output regression.

    Parameters:
      x_train (DataFrame): Preprocessed feature DataFrame (including 'ID')
      y_train (DataFrame): Target DataFrame with columns c01...c23.
      params (dict): Optional parameters for the RandomForestRegressor.

    Returns:
      model: Trained RandomForestRegressor.
    """
    if params is None:
        params = {
            "n_estimators": 5,
            "max_depth": 7,
            "min_samples_split": 0.01,
            "min_samples_leaf": 30,
            "random_state": 29,
            "n_jobs": -1,
        }
    model = RandomForestRegressor(**params)
    X_train = x_train.drop("ID", axis=1) if "ID" in x_train.columns else x_train
    model.fit(X_train, y_train)
    return model


def run_experiment(x_train, y_train, x_val, y_val, params=None):
    """
    Runs the full experiment: training the model and evaluating it on validation data.

    Parameters:
      x_train (DataFrame): Training features.
      y_train (DataFrame): Training targets.
      x_val (DataFrame): Validation features.
      y_val (DataFrame): Validation targets.

    Returns:
      model: Trained RandomForestRegressor.
    """
    model = train_random_forest(x_train, y_train, params=params)
    metric_train = evaluate_model(model, x_train, y_train)
    metric_val = evaluate_model(model, x_val, y_val)
    print("Training Weighted RMSE: {:.4f}".format(metric_train))
    print("Validation Weighted RMSE: {:.4f}".format(metric_val))
    return model


def benchmark_random_forest(
    x_train, y_train, x_val, y_val, param_grid=None, verbose=True, subset_humidity=False
):
    """Perform hyperparameter search for RandomForestRegressor and evaluate on humidity-based subsets.

    Parameters:
        x_train (DataFrame): Training features
        y_train (DataFrame): Training targets
        x_val (DataFrame): Validation features
        y_val (DataFrame): Validation targets
        verbose (bool): Whether to print progress and results
        subset_humidity (bool): Whether to evaluate on humidity-based subsets

    Returns:
        tuple: (best_model, best_params, best_val_rmse)
    """
    # Define parameter grid
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [7, 10, 15, None],
            "min_samples_split": [0.005, 0.01, 0.02],
            "min_samples_leaf": [10, 20, 30, 50],
        }

    # Create all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    # Add fixed parameters
    fixed_params = {"random_state": 29, "n_jobs": -1}
    for params in param_combinations:
        params.update(fixed_params)

    # Create humidity-based validation subsets
    if subset_humidity:
        humidity_subsets = create_humidity_subsets(x_val, y_val)

    best_val_rmse = float("inf")
    best_model = None
    best_params = None

    # Progress bar for parameter combinations
    if verbose:
        param_combinations = tqdm(param_combinations, desc="Hyperparameter Search")

    # Evaluate each parameter combination
    for params in param_combinations:
        model = train_random_forest(x_train, y_train, params=params)

        # Evaluate on full validation set
        train_rmse = evaluate_model(model, x_train, y_train)
        val_rmse = evaluate_model(model, x_val, y_val)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = params

        if verbose:
            print(f"\nParameters: {params}")
            print(f"Training Weighted RMSE: {train_rmse:.4f}")
            print(f"Validation Weighted RMSE: {val_rmse:.4f}")

            # Evaluate on humidity subsets
            if subset_humidity:
                print("\nPerformance on humidity subsets:")
                for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                    subset_rmse = evaluate_model(model, x_subset, y_subset)
                    humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                    print(
                        f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})"
                    )
            print("\n" + "-" * 50)

    if verbose:
        print("\nBest Model:")
        print(f"Parameters: {best_params}")
        print(f"Validation RMSE: {best_val_rmse:.4f}")

        # Evaluate best model on humidity subsets
        if subset_humidity:
            print("\nBest Model Performance on humidity subsets:")
            for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                subset_rmse = evaluate_model(best_model, x_subset, y_subset)
                humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                print(
                    f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})"
                )
        else:
            print("\nBest Model Performance on validation set:")
            print(f"Validation RMSE: {best_val_rmse:.4f}")

    return best_model, best_params, best_val_rmse
