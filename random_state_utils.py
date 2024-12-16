import numpy as np
from sklearn.ensemble import RandomForestRegressor
from model_utils import XGBRegressorWrapper

def get_random_states():
    """Generate 5 different random states for non-deterministic models."""
    return [42, 123, 456, 789, 101112]

def is_non_deterministic(model):
    """Check if a model is non-deterministic (Random Forest or XGBoost)."""
    return isinstance(model, (RandomForestRegressor, XGBRegressorWrapper))

def train_model_with_random_states(model_class, params, X_train, y_train, X_val, y_val, random_states):
    """Train model with multiple random states and return predictions and metrics.

    Args:
        model_class: The model class (RandomForestRegressor or XGBRegressorWrapper)
        params: Model parameters
        X_train, y_train: Training data
        X_val, y_val: Validation data
        random_states: List of random states to use

    Returns:
        List of predictions for each random state
    """
    predictions = []
    for seed in random_states:
        # Create a new model instance with the current random state
        if 'random_state' in params:
            params['random_state'] = seed
        model = model_class(**params)

        # Train the model
        model.fit(X_train, y_train)

        # Get predictions
        pred = model.predict(X_val)
        predictions.append(pred)

    return np.array(predictions)
