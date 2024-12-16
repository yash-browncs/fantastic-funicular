import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self, dummy=0):
        self.dummy = dummy

    def get_params(self, deep=True):
        return {'dummy': self.dummy}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X['Temperature (C)'].values

    def __getstate__(self):
        return {'dummy': self.dummy}

    def __setstate__(self, state):
        self.dummy = state['dummy']

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'random_state': self.random_state
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def select_best_parameters(validation_results, metric='r2'):
    param_scores = {}
    for result in validation_results:
        param_key = str(result['params'])
        if param_key not in param_scores:
            param_scores[param_key] = []
        param_scores[param_key].append(result[metric])

    mean_scores = {params: np.mean(scores) for params, scores in param_scores.items()}

    if metric == 'rmse':
        best_params_str = min(mean_scores.items(), key=lambda x: x[1])[0]
    else:
        best_params_str = max(mean_scores.items(), key=lambda x: x[1])[0]

    return eval(best_params_str)

def train_final_model(model_class, best_params, X_train, y_train, random_state=None):
    if random_state is not None:
        model = model_class(**best_params, random_state=random_state)
    else:
        model = model_class(**best_params)

    model.fit(X_train, y_train)
    return model

def aggregate_metrics(results, model_name=None):
    if model_name:
        results = [r for r in results if model_name in r['model']]

    if not results:
        return None

    rmse_scores = [r['rmse'] for r in results]
    r2_scores = [r['r2'] for r in results]

    return {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores)
    }
