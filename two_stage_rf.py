import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier  # , RandomForestRegressor
from xgboost import XGBRegressor
from utils import evaluate_model


class TwoStageRandomForest(BaseEstimator):
    """
    A scikit-learn compatible estimator for a two-stage prediction task.

    It wraps two separate models:
      - A multi-label RandomForestClassifier that predicts which columns are active.
      - A RandomForestRegressor that predicts the scalar value for the active columns.

    The fit method expects y to be a tuple: (y_classif, y_reg), where:
      - y_classif: Binary targets of shape (n_samples, n_outputs) for the classifier.
      - y_reg: Regression targets of shape (n_samples,) or (n_samples, 1) for the regressor.

    The predict method returns an array of shape (n_samples, n_outputs) where for each row,
    the regressor's prediction is assigned to the columns marked active by the classifier (and 0 otherwise).
    """

    def __init__(self, clf_params=None, reg_params=None):
        # Set default hyperparameters if none provided.
        self.clf_params = clf_params if clf_params is not None else {}
        self.reg_params = reg_params if reg_params is not None else {}
        self.clf = RandomForestClassifier(**self.clf_params)
        self.reg = XGBRegressor(**self.reg_params)
        # RandomForestRegressor(**self.reg_params)

    def fit(self, X, y):
        """
        Fit the two-stage model.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
        - y: tuple (y_classif, y_reg) where:
            y_classif: array-like of shape (n_samples, n_outputs) with binary targets.
            y_reg: array-like of shape (n_samples,) or (n_samples, 1) with regression targets.
        """
        if not (isinstance(y, (tuple, list)) and len(y) == 2):
            raise ValueError("y must be a tuple: (y_classif, y_reg)")

        y_classif, y_reg = y

        # Fit the classifier on the multi-label binary targets.
        self.clf.fit(X, y_classif)
        # Ensure y_reg is a 1D array.
        y_reg = np.ravel(y_reg)
        sample_weight = np.where(y_reg > 0.5, 1.2, 1.0)
        self.reg.fit(X, y_reg, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Predict using the two-stage model.

        First, predicts which columns are active using the classifier.
        Then, predicts the scalar value using the regressor.
        Finally, combines them: assigns the regressor's prediction to the columns marked active (1)
        and 0 to the inactive ones.

        Returns:
        - y_pred: array of shape (n_samples, n_outputs)
        """
        # Predict active/inactive columns (binary mask)
        y_classif_pred = self.clf.predict(X)
        # Predict the regression value for each sample
        y_reg_pred = self.reg.predict(X)

        # Ensure y_classif_pred is 2D (if only one target, reshape accordingly)
        if y_classif_pred.ndim == 1:
            y_classif_pred = y_classif_pred.reshape(-1, 1)

        # Combine the predictions: multiply binary mask with regressor's prediction broadcast across columns.
        y_pred = y_classif_pred * y_reg_pred[:, np.newaxis]
        return y_pred


def train_two_stage_rf(x_train, y_train, clf_params=None, reg_params=None):
    """
    Train a two-stage RandomForest model.

    Parameters:
    - x_train: array-like of shape (n_samples, n_features)
    - y_train: tuple (y_classif, y_reg) where:
        y_classif: array-like of shape (n_samples, n_outputs) with binary targets.
        y_reg: array-like of shape (n_samples,) or (n_samples, 1) with regression targets.
    - clf_params: dict, optional, parameters for the RandomForestClassifier.
    - reg_params: dict, optional, parameters for the RandomForestRegressor.
    Returns:
    - model: Trained TwoStageRandomForest model.
    """
    if clf_params is None:
        clf_params = {
            "n_estimators": 5,
            "max_depth": 7,
            "min_samples_split": 0.01,
            "min_samples_leaf": 30,
            "random_state": 29,
            "n_jobs": -1,
        }
    if reg_params is None:
        reg_params = {
            "n_estimators": 5,
            "max_depth": 7,
            "min_samples_split": 0.01,
            "min_samples_leaf": 30,
            "random_state": 29,
            "n_jobs": -1,
        }
    model = TwoStageRandomForest(clf_params=clf_params, reg_params=reg_params)
    X_train = x_train.drop("ID", axis=1) if "ID" in x_train.columns else x_train
    model.fit(X_train, y_train)
    return model


def run_experiment(
    x_train,
    y_train,
    y_train_classif,
    y_train_reg,
    x_val,
    y_val,
    y_val_classif,
    y_val_reg,
    clf_params=None,
    reg_params=None,
):
    """
    Runs the full experiment: training the model and evaluating it on validation data.
    Parameters:
      x_train (DataFrame): Training features.
      y_train (DataFrame): Training targets.
      y_train_classif (DataFrame): Training targets for the classifier.
      y_train_reg (DataFrame): Training targets for the regressor.
      x_val (DataFrame): Validation features.
      y_val (DataFrame): Validation targets (tuple with binary and regression targets).
      y_val_classif (DataFrame): Validation targets for the classifier.
      y_val_reg (DataFrame): Validation targets for the regressor.
      clf_params (dict): Optional parameters for the RandomForestClassifier.
      reg_params (dict): Optional parameters for the RandomForestRegressor.
    Returns:
      model: Trained TwoStageRandomForest model.
    """
    model = train_two_stage_rf(
        x_train,
        (y_train_classif, y_train_reg),
        clf_params=clf_params,
        reg_params=reg_params,
    )
    metric_train = evaluate_model(model, x_train, y_train)
    metric_val = evaluate_model(model, x_val, y_val)
    print(f"Training Weighted RMSE: {metric_train:.4f}")
    print(f"Validation Weighted RMSE: {metric_val:.4f}")
    # compute accuracy of multilabel classifier
    y_pred = model.clf.predict(
        x_val.drop("ID", axis=1) if "ID" in x_val.columns else x_val
    )
    accuracy = np.mean(y_pred == y_val_classif)
    print(f"Validation Accuracy of Classifier: {accuracy:.4f}")

    return model
