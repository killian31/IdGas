from sklearn.ensemble import RandomForestRegressor

from utils import evaluate_model


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
