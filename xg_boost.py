import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from tqdm import tqdm
import itertools
from utils import evaluate_model, create_humidity_subsets


def train_xgboost(x_train, y_train, params=None):
    """
    Train a multi-output XGBoost model using MultiOutputRegressor.

    Parameters:
      x_train (DataFrame): Preprocessed training features (with 'ID' column dropped before training).
      y_train (DataFrame): Target values with columns c01...c23.
      params (dict): Optional parameters for XGBRegressor.

    Returns:
      model: Trained MultiOutputRegressor with XGBRegressor as base estimator.
    """
    if params is None:
        params = {
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "random_state": 29,
            "n_jobs": -1,  # Limit parallel jobs
            "tree_method": "approx",  # More memory efficient than 'hist'
            "max_bin": 64,  # Reduced from 128
            "scale_pos_weight": 1,
            "process_type": "default",
            "predictor": "cpu_predictor",
            "grow_policy": "depthwise",  # Changed from 'lossguide' for better memory usage
        }
    base_estimator = xgb.XGBRegressor(**params)
    model = MultiOutputRegressor(base_estimator)
    X_train = x_train.drop("ID", axis=1) if "ID" in x_train.columns else x_train
    sample_weight = np.mean(np.where(y_train >= 0.5, 1.2, 1), axis=1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def run_experiment(x_train, y_train, x_val, y_val, params=None):
    """
    Runs the full experiment with XGBoost:
      - Trains the model
      - Evaluates it on validation data
      - Prints and returns the model.

    Parameters:
      x_train (DataFrame): Training features.
      y_train (DataFrame): Training targets.
      x_val (DataFrame): Validation features.
      y_val (DataFrame): Validation targets.

    Returns:
      model: Trained MultiOutputRegressor with XGBRegressor.
    """
    model = train_xgboost(x_train, y_train, params=params)
    metric_train = evaluate_model(model, x_train, y_train)
    metric_val = evaluate_model(model, x_val, y_val)
    print("Training Weighted RMSE (XGBoost): {:.4f}".format(metric_train))
    print("Validation Weighted RMSE (XGBoost): {:.4f}".format(metric_val))
    return model


def benchmark_xgboost(
    x_train, y_train, x_val, y_val, param_grid=None, verbose=True, subset_humidity=False
):
    """Perform hyperparameter search for XGBoost and evaluate on humidity-based subsets.

    Parameters:
        x_train (DataFrame): Training features
        y_train (DataFrame): Training targets
        x_val (DataFrame): Validation features
        y_val (DataFrame): Validation targets
        param_grid (dict): Dictionary of parameter grids for XGBoost
        verbose (bool): Whether to print progress and results
        subset_humidity (bool): Whether to evaluate on humidity-based subsets

    Returns:
        tuple: (best_model, best_params, best_val_rmse)
    """
    # Define parameter grid
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [6, 8, 12],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [10, 30, 50],
            "gamma": [0, 0.1],
            "reg_alpha": [0, 0.01],
            "reg_lambda": [1, 5],
        }

    # Create all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    # Add fixed parameters
    fixed_params = {
        "random_state": 29,
        "n_jobs": -1,
        "tree_method": "approx",
        "max_bin": 64,
        "scale_pos_weight": 1,
        "process_type": "default",
        "predictor": "cpu_predictor",
        "grow_policy": "depthwise",
    }
    for params in param_combinations:
        params.update(fixed_params)

    # Create humidity-based validation subsets
    if subset_humidity:
        humidity_subsets = create_humidity_subsets(x_val, y_val)

    best_val_rmse = float("inf")
    best_model = None
    best_params = None

    try:
        # Progress bar for parameter combinations
        pbar = tqdm(total=len(param_combinations), desc="Training XGBoost")

        # Evaluate each parameter combination
        for params in param_combinations:
            model = train_xgboost(x_train, y_train, params=params)

            # Evaluate on full validation set
            train_rmse = evaluate_model(model, x_train, y_train)
            val_rmse = evaluate_model(model, x_val, y_val)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model
                best_params = params
            pbar.update(1)
            if verbose:
                print(f"\nParameters: {params}")

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
            pbar.set_description(
                f"Train WRMSE: {train_rmse:.4f} Val WRMSE: {val_rmse:.4f}"
            )

        print("\nBest XGBoost Model:")
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
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    finally:
        pbar.close()
        return best_model, best_params, best_val_rmse
