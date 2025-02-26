import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from utils import evaluate_model


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
            "n_jobs": 4,  # Limit parallel jobs
            "tree_method": "approx",  # More memory efficient than 'hist'
            "max_bin": 64,  # Reduced from 128
            "scale_pos_weight": 1,
            "process_type": "default",
            "predictor": "cpu_predictor",
            "grow_policy": "depthwise"  # Changed from 'lossguide' for better memory usage
        }
    base_estimator = xgb.XGBRegressor(**params)
    model = MultiOutputRegressor(base_estimator)
    X_train = x_train.drop("ID", axis=1) if "ID" in x_train.columns else x_train
    model.fit(X_train, y_train)
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
    metric = evaluate_model(model, x_val, y_val)
    print("Validation Weighted RMSE (XGBoost): {:.4f}".format(metric))
    return model
