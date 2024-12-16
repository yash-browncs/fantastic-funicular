import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import warnings
from datetime import datetime
from model_utils import XGBRegressorWrapper, BaselineModel
from random_state_utils import get_random_states, is_non_deterministic, train_model_with_random_states
warnings.filterwarnings('ignore')

def log_progress(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def create_walk_forward_windows(df):
    # Get last year start date
    last_year_start = df['Formatted Date'].max() - pd.Timedelta(days=365)

    # Create windows only within training period
    windows = []
    start_date = df['Formatted Date'].min()
    train_window = pd.Timedelta(days=365.25 * 5)  # 5 years
    val_window = pd.Timedelta(days=365.25/2)      # 6 months
    shift = pd.Timedelta(days=365.25/2)           # 6 months

    current_start = start_date
    while current_start + train_window + val_window <= last_year_start:
        train_end = current_start + train_window
        val_end = train_end + val_window

        windows.append({
            'train': (current_start, train_end),
            'val': (train_end, val_end)
        })
        current_start += shift

    return windows, last_year_start

def verify_windows(windows):
    print(f"\nNumber of windows created: {len(windows)}")
    if len(windows) > 0:
        first_window = windows[0]
        last_window = windows[-1]
        print("\nFirst window:")
        for split, (start, end) in first_window.items():
            print(f"{split}: {start} to {end}")
        print("\nLast window:")
        for split, (start, end) in last_window.items():
            print(f"{split}: {start} to {end}")

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def main():
    log_progress("Loading data...")
    df = pd.read_csv('weatherHistory.csv')
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['Next_Temperature'] = df['Temperature (C)'].shift(-1)
    df = df.dropna(subset=['Next_Temperature'])

    # Create windows for walk-forward validation and get test set cutoff
    windows, last_year_start = create_walk_forward_windows(df)
    verify_windows(windows)

    # Split final test set
    test_mask = df['Formatted Date'] >= last_year_start
    final_test_data = df[test_mask]
    train_val_data = df[~test_mask]

    # Parameter distributions for model tuning
    param_distributions = {
        'ridge': {
            'alpha': uniform(0.01, 10.0)
        },
        'lasso': {
            'alpha': uniform(0.01, 10.0)
        },
        'rf': {
            'n_estimators': [5],
            'max_depth': randint(3, 10),
            'min_samples_split': [2, 4]
        },
        'svr': {
            'C': [0.1],
            'gamma': ['scale']
        },
        'xgb': {
            'n_estimators': [5],  # Changed from 50 to 10
            'max_depth': [3, 6],
            'learning_rate': [0.1],
            'subsample': [0.8]
        },
        'baseline': {'dummy': [0]}  # Dummy parameter for baseline model
    }

    # Model initialization
    models = {
        'ridge': Ridge(),
        'lasso': Lasso(),
        'rf': RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=1,
        ),
        'svr': SVR(
            kernel='rbf',
            max_iter=1000
        ),
        'xgb': XGBRegressorWrapper(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8
        )
    }

    models['baseline'] = BaselineModel()

    # Initialize storage for validation results
    validation_results = {model_name: [] for model_name in ['ridge', 'lasso', 'rf', 'svr', 'xgb', 'baseline']}
    random_states = get_random_states()

    print("\n=== Training Models on Validation Windows ===")
    for window_idx, window in enumerate(windows, 1):
        log_progress(f"\nProcessing window {window_idx}/{len(windows)}")

        # Split data according to window
        train_mask = (df['Formatted Date'] >= window['train'][0]) & (df['Formatted Date'] < window['train'][1])
        val_mask = (df['Formatted Date'] >= window['val'][0]) & (df['Formatted Date'] < window['val'][1])

        # Prepare features and target
        features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
        X = df[features]
        y = df['Next_Temperature']

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Create subset for parameter search
        subset_size = int(0.05 * len(X_train))
        subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train.iloc[subset_idx]
        y_train_subset = y_train.iloc[subset_idx]

        for name, model in models.items():
            log_progress(f"Training {name} on window {window_idx}...")

            try:
                # Set up data for parameter search
                if name == 'svr':
                    search_X = X_train_subset[:int(len(X_train_subset)*0.2)]
                    search_y = y_train_subset[:int(len(y_train_subset)*0.2)]
                elif name in ['ridge', 'lasso']:
                    search_X = X_train
                    search_y = y_train
                else:
                    search_X = X_train_subset
                    search_y = y_train_subset

                # Perform parameter search
                random_search = RandomizedSearchCV(
                    model,
                    param_distributions[name],
                    n_iter=3 if name == 'svr' else 5,
                    cv=2,  # Changed from 1 to 2 as required by scikit-learn
                    scoring='neg_mean_squared_error',
                    random_state=42,
                    n_jobs=-1,
                    verbose=2
                )
                random_search.fit(search_X, search_y)

                # Train and evaluate on validation set
                if is_non_deterministic(model):
                    # For RF and XGB, evaluate with multiple random states
                    for rs in random_states:
                        model_instance = type(model)(**random_search.best_params_, random_state=rs)
                        model_instance.fit(X_train, y_train)
                        val_pred = model_instance.predict(X_val)
                        metrics = evaluate_model(y_val, val_pred, f'{name}_Window_{window_idx}_RS_{rs}')
                        metrics.update({
                            'window': window_idx,
                            'random_state': rs,
                            'params': random_search.best_params_
                        })
                        validation_results[name].append(metrics)
                else:
                    # For deterministic models, use single evaluation
                    model_instance = random_search.best_estimator_
                    val_pred = model_instance.predict(X_val)
                    metrics = evaluate_model(y_val, val_pred, f'{name}_Window_{window_idx}')
                    metrics.update({
                        'window': window_idx,
                        'params': random_search.best_params_
                    })
                    validation_results[name].append(metrics)

            except Exception as e:
                log_progress(f"Error training {name} on window {window_idx}: {str(e)}")
                continue

    # Select best parameters based on aggregated validation metrics
    best_params = {}
    final_models = {}
    final_results = []

    print("\n=== Training Final Models with Best Parameters ===")
    # Skip SVR model due to poor performance and long training time
    model_names = [name for name in models.keys() if name != 'svr']
    for name in model_names:
        try:
            log_progress(f"Processing {name} model...")
            # Get best parameters based on mean validation performance
            model_metrics = validation_results[name]
            if not model_metrics:
                log_progress(f"Skipping {name} - no validation metrics available")
                continue

            # Group metrics by parameters
            param_scores = {}
            for metric in model_metrics:
                param_str = str(metric['params'])
                if param_str not in param_scores:
                    param_scores[param_str] = []
                param_scores[param_str].append(metric['r2'])

            # Select parameters with best mean score
            mean_scores = {params: np.mean(scores) for params, scores in param_scores.items()}
            log_progress(f"Mean scores for {name}: {mean_scores}")

            best_param_str = max(mean_scores.items(), key=lambda x: x[1])[0]
            log_progress(f"Best parameters string for {name}: {best_param_str}")

            # Safely evaluate parameter string
            try:
                params_dict = {}
                for param_pair in best_param_str.strip('{}').split(','):
                    if param_pair.strip():
                        key, value = param_pair.split(':')
                        key = key.strip().strip("'")
                        value = value.strip()
                        # Handle numpy float64 values
                        if 'np.float64' in value:
                            value = float(value.split('(')[1].split(')')[0])
                        # Convert other string values to appropriate types
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'none':
                            value = None
                        elif value.replace('.', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                        elif value.startswith("'") and value.endswith("'"):
                            value = value.strip("'")
                        params_dict[key] = value
                best_params[name] = params_dict
            except Exception as e:
                log_progress(f"Error parsing parameters for {name}: {str(e)}")
                continue

            log_progress(f"Parsed parameters for {name}: {best_params[name]}")

            # Train final model(s) on all training data
            X_train = train_val_data[features]
            y_train = train_val_data['Next_Temperature']
            X_test = final_test_data[features]
            y_test = final_test_data['Next_Temperature']

            if is_non_deterministic(models[name]):
                for rs in random_states:
                    log_progress(f"Training {name} with random_state {rs}")
                    model_instance = type(models[name])(**best_params[name], random_state=rs)
                    model_instance.fit(X_train, y_train)
                    test_pred = model_instance.predict(X_test)
                    metrics = evaluate_model(y_test, test_pred, f'{name}_Final_RS_{rs}')
                    metrics['random_state'] = rs
                    final_results.append(metrics)
                    final_models[f'{name}_RS_{rs}'] = model_instance
            else:
                log_progress(f"Training final {name} model")
                model_instance = type(models[name])(**best_params[name])
                model_instance.fit(X_train, y_train)
                test_pred = model_instance.predict(X_test)
                metrics = evaluate_model(y_test, test_pred, f'{name}_Final')
                final_results.append(metrics)
                final_models[name] = model_instance

        except Exception as e:
            log_progress(f"Error processing {name}: {str(e)}")
            continue

    # Calculate baseline performance on test set
    log_progress("Calculating baseline performance")
    baseline_pred = X_test['Temperature (C)'].values
    baseline_metrics = evaluate_model(y_test, baseline_pred, 'Baseline_Final')
    final_results.append(baseline_metrics)

    # Print final results
    print("\n=== Final Test Set Results ===")
    for name in models.keys():
        model_results = [r for r in final_results if name in r['model']]
        if model_results:
            rmse_scores = [r['rmse'] for r in model_results]
            r2_scores = [r['r2'] for r in model_results]
            print(f"\n{name}:")
            print(f"RMSE - Mean: {np.mean(rmse_scores):.4f}, Std: {np.std(rmse_scores):.4f}")
            print(f"R2   - Mean: {np.mean(r2_scores):.4f}, Std: {np.std(r2_scores):.4f}")

    # Save results
    log_progress("Saving results and models...")
    results_dict = {
        'validation_results': validation_results,
        'best_params': best_params,
        'final_results': final_results,
        'final_models': final_models,
        'test_windows': windows,
    }

    try:
        with open('model_results.pkl', 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("\nResults and models saved to 'model_results.pkl'")
    except Exception as e:
        print(f"\nError saving results: {str(e)}")

if __name__ == "__main__":
    main()
