import numpy as np
from sklearn.base import BaseEstimator
from utils import evaluate_model
import itertools
from tqdm import tqdm

# --- Import Classifiers ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Import Regressors ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier

def get_classifier_instance(name: str, params: dict):
    """
    Returns an instance of a classifier given its name and parameters.
    """
    name = name.lower()
    if name in ['randomforestclassifier', 'rf']:
        return RandomForestClassifier(**params)
    elif name in ['decisiontreeclassifier', 'dt']:
        return DecisionTreeClassifier(**params)
    elif name in ['kneighborsclassifier', 'knn']:
        return KNeighborsClassifier(**params)
    elif name in ['xgbclassifier', 'xgb']:
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Classifier '{name}' is not supported. Please choose from: "
                         "'randomforestclassifier', 'logisticregression', 'decisiontreeclassifier', "
                         "'svc', 'kneighborsclassifier', 'gradientboostingclassifier', 'xgbclassifier'.")

def get_regressor_instance(name: str, params: dict):
    """
    Returns an instance of a regressor given its name and parameters.
    """
    name = name.lower()
    if name in ['randomforestregressor', 'rf']:
        return RandomForestRegressor(**params)
    elif name in ['linearregression', 'lr']:
        return LinearRegression(**params)
    elif name in ['decisiontreeregressor', 'dt']:
        return DecisionTreeRegressor(**params)
    elif name in ['svr']:
        return SVR(**params)
    elif name in ['kneighborsregressor', 'knn']:
        return KNeighborsRegressor(**params)
    elif name in ['xgbregressor', 'xgb']:
        return XGBRegressor(**params)
    elif name in ['gradientboostingregressor', 'gbr']:
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Regressor '{name}' is not supported. Please choose from: "
                         "'randomforestregressor', 'linearregression', 'decisiontreeregressor', "
                         "'svr', 'kneighborsregressor', 'xgbregressor', 'gradientboostingregressor'.")

class TwoStageModel(BaseEstimator):
    """
    A scikit-learn compatible estimator for a two-stage prediction task.

    This model wraps two separate components:
      - A classifier (selected via a string) that predicts which columns are active.
      - A regressor (selected via a string) that predicts the scalar value for the active columns.

    The fit method expects y to be a tuple: (y_classif, y_reg), where:
      - y_classif: Binary targets of shape (n_samples, n_outputs) for the classifier.
      - y_reg: Regression targets of shape (n_samples,) or (n_samples, 1) for the regressor.

    The predict method returns an array of shape (n_samples, n_outputs) where for each row,
    the regressor's prediction is assigned to the columns marked active by the classifier (and 0 otherwise).
    """
    def __init__(self, 
                 clf_name='randomforestclassifier', 
                 reg_name='xgbregressor', 
                 clf_params=None, 
                 reg_params=None):
        self.clf_name = clf_name
        self.reg_name = reg_name
        self.clf_params = clf_params if clf_params is not None else {}
        self.reg_params = reg_params if reg_params is not None else {}
        self.clf = get_classifier_instance(self.clf_name, self.clf_params)
        self.reg = get_regressor_instance(self.reg_name, self.reg_params)

    def fit(self, X, y):
        """
        Fit the two-stage model.
        
        Parameters:
          X: array-like of shape (n_samples, n_features)
          y: tuple (y_classif, y_reg) where:
              y_classif: array-like of shape (n_samples, n_outputs) with binary targets.
              y_reg: array-like of shape (n_samples,) or (n_samples, 1) with regression targets.
        """
        if not (isinstance(y, (tuple, list)) and len(y) == 2):
            raise ValueError("y must be a tuple: (y_classif, y_reg)")
        y_classif, y_reg = y

        # Fit classifier on binary targets.
        self.clf.fit(X, y_classif)
        # Ensure y_reg is 1D.
        y_reg = np.ravel(y_reg)
        # Apply sample weighting
        if not isinstance(self.reg, KNeighborsRegressor) and not isinstance(self.reg, SVR):
            sample_weight = np.where(y_reg > 0.5, 1.2, 1.0)
            self.reg.fit(X, y_reg, sample_weight=sample_weight)
        else:
            self.reg.fit(X, y_reg)
        return self

    def predict(self, X):
        """
        Predict using the two-stage model.
        
        First, the classifier predicts which columns are active.
        Then, the regressor predicts the scalar value.
        The final prediction assigns the regressor's value to active columns (1)
        and 0 to inactive columns.
        
        Returns:
          y_pred: array of shape (n_samples, n_outputs)
        """
        y_classif_pred = self.clf.predict(X)
        y_reg_pred = self.reg.predict(X)

        # Reshape classifier predictions if needed.
        if y_classif_pred.ndim == 1:
            y_classif_pred = y_classif_pred.reshape(-1, 1)
        y_pred = y_classif_pred * y_reg_pred[:, np.newaxis]
        return y_pred


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
        self.clf = XGBClassifier(**self.clf_params)
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


def train_two_stage_model(x_train, y_train, 
                          clf_name='randomforestclassifier', 
                          reg_name='xgbregressor', 
                          clf_params=None, 
                          reg_params=None):
    """
    Train the two-stage model.
    
    Parameters:
      x_train: DataFrame or array-like of shape (n_samples, n_features)
      y_train: tuple (y_classif, y_reg) with training targets.
      clf_name: String name of the classifier to use.
      reg_name: String name of the regressor to use.
      clf_params: Optional dict of parameters for the classifier.
      reg_params: Optional dict of parameters for the regressor.
      
    Returns:
      model: Trained TwoStageModel instance.
    """
    # Default parameters if none are provided.
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
    model = TwoStageModel(clf_name=clf_name, 
                          reg_name=reg_name, 
                          clf_params=clf_params, 
                          reg_params=reg_params)
    # Optionally remove an "ID" column if present.
    X_train = x_train.drop("ID", axis=1) if hasattr(x_train, "columns") and "ID" in x_train.columns else x_train
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
    clf_name='randomforestclassifier',
    reg_name='xgbregressor',
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
      clf_name (str): String name for the classifier.
      reg_name (str): String name for the regressor.
      clf_params (dict): Optional parameters for the classifier.
      reg_params (dict): Optional parameters for the regressor.
      
    Returns:
      model: Trained TwoStageModel instance.
    """
    model = train_two_stage_model(
        x_train,
        (y_train_classif, y_train_reg),
        clf_name=clf_name,
        reg_name=reg_name,
        clf_params=clf_params,
        reg_params=reg_params,
    )
    # Evaluate the model (assuming evaluate_model is defined elsewhere)
    metric_train = evaluate_model(model, x_train, y_train)
    metric_val = evaluate_model(model, x_val, y_val)
    print(f"Training Weighted RMSE: {metric_train:.4f}")
    print(f"Validation Weighted RMSE: {metric_val:.4f}")
    
    X_val_features = x_val.drop("ID", axis=1) if hasattr(x_val, "columns") and "ID" in x_val.columns else x_val
    y_pred = model.clf.predict(X_val_features)
    accuracy = np.mean(y_pred == y_val_classif)
    print(f"Validation Accuracy of Classifier: {accuracy:.4f}")

    return model


def benchmark_2s_model_grid(x_train, y_train, y_train_classif, y_train_reg, x_val, y_val, y_val_classif, y_val_reg, model_grid=None, verbose=True, subset_humidity=False):
    """
    Benchmark all combinations of classifier and regressor specified in model_grid.

    Parameters:
        x_train (DataFrame): Training features.
        y_train (DataFrame): Training targets.
        y_train_classif (DataFrame): Training targets for the classifier.
        y_train_reg (DataFrame): Training targets for the regressor.
        x_val (DataFrame): Validation features.
        y_val (DatFrame): Validation targets.
        y_val_classif (DataFrame): Validation targets for the classifier.
        y_val_reg (DataFrame): Validation targets for the regressor.
        model_grid (dict): Dictionary with keys:
            - "clf_names": list of classifier names (strings).
            - "reg_names": list of regressor names (strings).
            Optionally, you can also pass:
            - "clf_params": dict mapping classifier names to their parameter dictionaries.
            - "reg_params": dict mapping regressor names to their parameter dictionaries.
            If not provided, empty parameter dicts are used.
        verbose (bool): Whether to print progress and results.
        subset_humidity (bool): If True, evaluate on humidity-based subsets (using create_humidity_subsets).

    Returns:
        tuple: (best_model, best_params, best_val_rmse)
            best_params is a dict with keys: "clf_name", "reg_name", "clf_params", "reg_params".
    """
    # Define a default model grid if none is provided.
    if model_grid is None:
        model_grid = {
            "clf_names": ["randomforestclassifier", "xgbclassifier"],
            "reg_names": ["xgbregressor", "randomforestregressor"],
        }
    # Use provided parameter dicts or default to empty dicts.
    clf_params_grid = model_grid.get("clf_params", {name: {} for name in model_grid["clf_names"]})
    reg_params_grid = model_grid.get("reg_params", {name: {} for name in model_grid["reg_names"]})

    best_val_rmse = float("inf")
    best_model = None
    best_params = None
    best_accuracy = 0

    # Create humidity-based validation subsets if requested.
    if subset_humidity:
        humidity_subsets = create_humidity_subsets(x_val, y_val)

    # Create all model combinations.
    model_combinations = list(itertools.product(model_grid["clf_names"], model_grid["reg_names"]))
    model_combinations = tqdm(model_combinations, desc="Model Grid Search")

    for clf_name, reg_name in model_combinations:
        clf_params = clf_params_grid.get(clf_name, {})
        reg_params = reg_params_grid.get(reg_name, {})

        # Instantiate and train the two-stage model.
        model = train_two_stage_model(x_train, (y_train_classif, y_train_reg), clf_name, reg_name, clf_params, reg_params)
        train_rmse = evaluate_model(model, x_train, y_train)
        val_rmse = evaluate_model(model, x_val, y_val)
        pred_val = model.clf.predict(x_val.drop("ID", axis=1))
        accuracy = np.mean(pred_val == y_val_classif)

        # Update best model if validation error improves.
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = {
                "clf_name": clf_name,
                "reg_name": reg_name,
                "clf_params": clf_params,
                "reg_params": reg_params,
            }
            best_accuracy = accuracy

        if verbose:
            print(f"\nModel Combination: Classifier '{clf_name}' | "
                  f"Regressor '{reg_name}'")
            print(f"Training Weighted RMSE: {train_rmse:.4f}")
            print(f"Validation Weighted RMSE: {val_rmse:.4f}")
            print(f"Validation Accuracy of Classifier: {accuracy:.4f}")
            # Evaluate on humidity subsets if requested.
            if subset_humidity:
                print("Performance on humidity subsets:")
                for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                    subset_rmse = evaluate_model(model, x_subset, y_subset)
                    humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                    print(f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})")
                    preds = model.clf.predict(x_subset.drop("ID", axis=1))
                    subset_accuracy = np.mean(preds == y_subset[0])
                    print(f"Humidity {humidity_range}: Accuracy = {subset_accuracy:.4f} (n={len(x_subset)})")
            print("-" * 50)

    if verbose:
        print("\nBest Model:")
        print(f"Parameters: {best_params}")
        print(f"Validation RMSE: {best_val_rmse:.4f}")
        print(f"Validation Accuracy of Classifier: {best_accuracy:.4f}")
        if subset_humidity:
            print("Best Model Performance on humidity subsets:")
            for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                subset_rmse = evaluate_model(best_model, x_subset, y_subset)
                humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                print(f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})")
                preds = best_model.clf.predict(x_subset.drop("ID", axis=1))
                subset_accuracy = np.mean(preds == y_subset[0])
                print(f"Humidity {humidity_range}: Accuracy = {subset_accuracy:.4f} (n={len(x_subset)})")

    return best_model, best_params, best_val_rmse


def benchmark_2s_param_grid(x_train, y_train, y_train_classif, y_train_reg, x_val, y_val, y_val_classif, y_val_reg,
                         clf_name, reg_name,
                         param_grid=None, verbose=True, subset_humidity=False):
    """
    Benchmark different hyperparameter combinations for a fixed classifier/regressor pair.

    Parameters:
        x_train (DataFrame): Training features.
        y_train (tuple): Training targets as (y_classif, y_reg).
        x_val (DataFrame): Validation features.
        y_val (tuple): Validation targets as (y_classif, y_reg).
        clf_name (str): The classifier name to use.
        reg_name (str): The regressor name to use.
        param_grid (dict): Dictionary with keys 'clf' and 'reg' where each maps to a dict of hyperparameters to search.
            Example:
                param_grid = {
                    "clf": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [7, 10, 15, None],
                        "min_samples_split": [0.005, 0.01, 0.02],
                        "min_samples_leaf": [10, 20, 30, 50],
                    },
                    "reg": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [7, 10, 15, None],
                        "min_samples_split": [0.005, 0.01, 0.02],
                        "min_samples_leaf": [10, 20, 30, 50],
                    }
                }
        verbose (bool): Whether to print progress and results.
        subset_humidity (bool): If True, evaluate on humidity-based subsets.

    Returns:
        tuple: (best_model, best_params, best_val_rmse)
            best_params is a dict with keys "clf_params" and "reg_params".
    """
    # Define default parameter grid if none provided.
    if param_grid is None:
        param_grid = {
            "clf": {
                "n_estimators": [50, 100, 200],
                "max_depth": [7, 10, 15],
                "min_samples_split": [0.005, 0.01, 0.02],
                "min_samples_leaf": [10, 20, 30, 50],
            },
            "reg": {
                "n_estimators": [50, 100, 200],
                "max_depth": [7, 10, 15],
                "min_samples_split": [0.005, 0.01, 0.02],
                "min_samples_leaf": [10, 20, 30, 50],
            }
        }

    # Create all combinations of parameters for classifier and regressor.
    clf_param_list = [
        dict(zip(param_grid["clf"].keys(), values))
        for values in itertools.product(*param_grid["clf"].values())
    ]
    reg_param_list = [
        dict(zip(param_grid["reg"].keys(), values))
        for values in itertools.product(*param_grid["reg"].values())
    ]

    # Add fixed parameters.
    fixed_params = {"random_state": 29, "n_jobs": -1} if (clf_name != "kneighborsclassifier" and reg_name != "kneighborsregressor") else {"n_jobs": -1}
    for params in clf_param_list:
        params.update(fixed_params)
    for params in reg_param_list:
        params.update(fixed_params)

    best_val_rmse = float("inf")
    best_model = None
    best_params = None
    best_accuracy = 0

    # Create humidity subsets if needed.
    if subset_humidity:
        humidity_subsets = create_humidity_subsets(x_val, y_val)

    # Combine classifier and regressor parameters.
    param_combinations = list(itertools.product(clf_param_list, reg_param_list))
    param_combinations = tqdm(param_combinations, desc="Hyperparameter Search")

    for clf_params, reg_params in param_combinations:
        # Instantiate and train the two-stage model.
        model = train_two_stage_model(x_train, (y_train_classif, y_train_reg), clf_name, reg_name, clf_params, reg_params)

        train_rmse = evaluate_model(model, x_train, y_train)
        val_rmse = evaluate_model(model, x_val, y_val)
        pred_val = model.clf.predict(x_val.drop("ID", axis=1))
        accuracy = np.mean(pred_val == y_val_classif)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_accuracy = accuracy
            best_model = model
            best_params = {"clf_params": clf_params, "reg_params": reg_params}

        if verbose:
            print(f"\nParameters:\n Classifier params: {clf_params} \n Regressor params: {reg_params}")
            print(f"Training Weighted RMSE: {train_rmse:.4f}")
            print(f"Validation Weighted RMSE: {val_rmse:.4f}")
            print(f"Validation Accuracy of Classifier: {accuracy:.4f}")

            if subset_humidity:
                print("Performance on humidity subsets:")
                for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                    subset_rmse = evaluate_model(model, x_subset, y_subset)
                    humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                    print(f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})")
                    preds = model.clf.predict(x_subset.drop("ID", axis=1))
                    subset_accuracy = np.mean(preds == y_subset[0])
                    print(f"Humidity {humidity_range}: Accuracy = {subset_accuracy:.4f} (n={len(x_subset)})")
            print("-" * 50)

    if verbose:
        print("\nBest Model:")
        print(f"Best Parameters: {best_params}")
        print(f"Validation RMSE: {best_val_rmse:.4f}")
        print(f"Validation Accuracy of Classifier: {best_accuracy:.4f}")
        if subset_humidity:
            print("Best Model Performance on humidity subsets:")
            for i, (x_subset, y_subset) in enumerate(humidity_subsets):
                subset_rmse = evaluate_model(best_model, x_subset, y_subset)
                humidity_range = f"[{i*0.2:.1f}, {(i+1)*0.2:.1f}]"
                print(f"Humidity {humidity_range}: RMSE = {subset_rmse:.4f} (n={len(x_subset)})")
                preds = best_model.clf.predict(x_subset.drop("ID", axis=1))
                subset_accuracy = np.mean(preds == y_subset[0])
                print(f"Humidity {humidity_range}: Accuracy = {subset_accuracy:.4f} (n={len(x_subset)})")

    return best_model, best_params, best_val_rmse