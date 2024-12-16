import pickle
import numpy as np
from datetime import datetime
import pandas as pd

def load_results():
    with open('model_results.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_results(results):
    # Check dates
    print("\n=== Date Range Analysis ===")
    dates = []
    for window in results['test_windows']:
        start = pd.to_datetime(window['test_start'])
        end = pd.to_datetime(window['test_end'])
        dates.append((start, end))
        print(f"Test window: {start} to {end}")

    print("\n=== Model Configuration Verification ===")
    # Verify XGBoost n_estimators and CV settings
    for model_name in results.keys():
        if model_name != 'test_windows' and isinstance(results[model_name], list):
            if model_name == 'xgb':
                n_estimators = set(r['params'].get('n_estimators', 'N/A') for r in results[model_name])
                print(f"XGBoost n_estimators values used: {sorted(list(n_estimators))}")

    print("\n=== Random States Analysis ===")
    # Check random states for non-deterministic models
    for model in ['rf', 'xgb']:
        if model in results:
            random_states = set()
            for result in results[model]:
                if 'random_state' in result['params']:
                    random_states.add(result['params']['random_state'])
            print(f"{model} random states used: {sorted(list(random_states))}")

    print("\n=== Best Parameters by Model ===")
    # Find best parameters for each model
    for model_name in results.keys():
        if model_name != 'test_windows' and isinstance(results[model_name], list):
            # Get best result based on R2 score
            best_result = max(results[model_name], key=lambda x: x['test_r2'])
            print(f"\n{model_name.upper()} Best Parameters:")
            print(f"Parameters: {best_result['params']}")
            print(f"Test R2: {best_result['test_r2']:.4f}")
            print(f"Test RMSE: {best_result['test_rmse']:.4f}")

    print("\n=== Model Performance Summary ===")
    # Analyze performance metrics
    for model_name in results.keys():
        if model_name != 'test_windows' and isinstance(results[model_name], list):
            scores = {
                'rmse': [r['test_rmse'] for r in results[model_name]],
                'r2': [r['test_r2'] for r in results[model_name]]
            }
            print(f"\n{model_name.upper()} Statistics:")
            print(f"Number of evaluations: {len(scores['rmse'])}")
            print(f"RMSE - Mean: {np.mean(scores['rmse']):.4f}, Std: {np.std(scores['rmse']):.4f}")
            print(f"R2   - Mean: {np.mean(scores['r2']):.4f}, Std: {np.std(scores['r2']):.4f}")

if __name__ == "__main__":
    results = load_results()
    analyze_results(results)
