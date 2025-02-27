import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

from utils import evaluate_model


class SelectiveLinearRegressor(BaseEstimator, RegressorMixin):
    """
    A custom regressor that performs feature selection based on statistical significance
    and then fits a linear regression model using only significant features.
    
    This model performs the following steps for each target variable:
    1. Fits an initial linear regression model using statsmodels
    2. Identifies statistically significant features (p-value < alpha)
    3. Refits the model using only significant features
    4. Stores the final model and selected features for prediction
    """
    
    def __init__(self, alpha=0.05):
        """
        Initialize the SelectiveLinearRegressor.
        
        Parameters:
          alpha (float): Significance threshold for feature selection (default: 0.05)
        """
        self.alpha = alpha
        self.models = []
        self.selected_features = []
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Fit a separate linear regression model for each target variable,
        performing feature selection for each model.
        
        Parameters:
          X (array-like): Feature matrix
          y (array-like): Target matrix with multiple columns
          
        Returns:
          self: The fitted regressor
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
            
        # Add constant for statsmodels
        X_with_const = sm.add_constant(X_array)
        
        # Convert y to numpy array if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y_array = y.values
        else:
            y_array = y
            
        # Fit a model for each target variable
        for i in range(y_array.shape[1]):
            # Get the current target
            y_i = y_array[:, i]
            
            # Fit initial model with all features
            initial_model = sm.OLS(y_i, X_with_const).fit()
            
            # Get p-values and identify significant features
            p_values = initial_model.pvalues[1:]  # Skip the constant term
            significant_indices = np.where(p_values < self.alpha)[0]
            
            # If no features are significant, keep the one with the lowest p-value
            if len(significant_indices) == 0:
                significant_indices = [np.argmin(p_values)]
                
            # Store the indices of selected features
            self.selected_features.append(significant_indices)
            
            # Create a new feature matrix with only significant features
            X_significant = X_array[:, significant_indices]
            
            # Fit final model with significant features
            final_model = LinearRegression()
            final_model.fit(X_significant, y_i)
            
            # Store the final model
            self.models.append(final_model)
            
        return self
    
    def predict(self, X):
        """
        Generate predictions using the fitted models.
        
        Parameters:
          X (array-like): Feature matrix
          
        Returns:
          array: Predictions for all target variables
        """
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Initialize predictions array
        predictions = np.zeros((X_array.shape[0], len(self.models)))
        
        # Generate predictions for each target variable
        for i, (model, selected) in enumerate(zip(self.models, self.selected_features)):
            X_selected = X_array[:, selected]
            predictions[:, i] = model.predict(X_selected)
            
        return predictions
    
    def get_selected_features(self):
        """
        Get the names of selected features for each target variable.
        
        Returns:
          list: List of lists containing selected feature names for each target
        """
        selected_feature_names = []
        for indices in self.selected_features:
            selected_names = [self.feature_names[i] for i in indices]
            selected_feature_names.append(selected_names)
        return selected_feature_names


def train_linear_regression(x_train, y_train, alpha=0.05):
    """
    Train a SelectiveLinearRegressor model that performs feature selection
    based on statistical significance for each target variable.
    
    Parameters:
      x_train (DataFrame): Preprocessed training features
      y_train (DataFrame): Target values with columns c01...c23
      alpha (float): Significance threshold for feature selection (default: 0.05)
      
    Returns:
      model: Trained SelectiveLinearRegressor
    """
    # Drop ID column if present
    X_train = x_train.drop("ID", axis=1) if "ID" in x_train.columns else x_train
    
    # Initialize and fit the model
    model = SelectiveLinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Print information about selected features
    selected_features = model.get_selected_features()
    target_cols = y_train.columns if isinstance(y_train, pd.DataFrame) else [f"target_{i}" for i in range(y_train.shape[1])]
    
    print("Feature selection results:")
    for i, (target, features) in enumerate(zip(target_cols, selected_features)):
        print(f"{target}: {len(features)} features selected - {', '.join(features)}")
    
    return model


def run_experiment(x_train, y_train, x_val, y_val, alpha=0.05):
    """
    Runs the full experiment with SelectiveLinearRegressor:
      - Trains the model with feature selection
      - Evaluates it on training and validation data
      - Prints and returns the model
    
    Parameters:
      x_train (DataFrame): Training features
      y_train (DataFrame): Training targets
      x_val (DataFrame): Validation features
      y_val (DataFrame): Validation targets
      alpha (float): Significance threshold for feature selection (default: 0.05)
      
    Returns:
      model: Trained SelectiveLinearRegressor
    """
    print("Training SelectiveLinearRegressor model...")
    model = train_linear_regression(x_train, y_train, alpha=alpha)
    
    # Evaluate the model
    metric_train = evaluate_model(model, x_train, y_train)
    metric_val = evaluate_model(model, x_val, y_val)
    
    print("Training Weighted RMSE: {:.4f}".format(metric_train))
    print("Validation Weighted RMSE: {:.4f}".format(metric_val))
    
    return model