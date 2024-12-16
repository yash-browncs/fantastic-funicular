import pandas as pd
import pickle
import numpy as np
from model_utils import XGBRegressorWrapper, aggregate_metrics
from sklearn.ensemble import RandomForestRegressor
from random_state_utils import is_non_deterministic

def load_results():
    """Load saved model results"""
    with open('model_results.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_validation_statistics(validation_results):
    """Calculate statistics for validation results across windows."""
    validation_stats = {}

    for model_name, model_results in validation_results.items():
        if not model_results:
            continue

        # Calculate window-specific statistics
        window_stats = {}
        windows = sorted(set(r['window'] for r in model_results))

        for window in windows:
            window_results = [r for r in model_results if r['window'] == window]
            if window_results:
                window_stats[f'window_{window}'] = {
                    'rmse_mean': np.mean([r['rmse'] for r in window_results]),
                    'rmse_std': np.std([r['rmse'] for r in window_results]) if len(window_results) > 1 else 0.0,
                    'r2_mean': np.mean([r['r2'] for r in window_results]),
                    'r2_std': np.std([r['r2'] for r in window_results]) if len(window_results) > 1 else 0.0,
                    'n_scores': len(window_results)
                }

        # Calculate overall statistics
        all_rmse = [r['rmse'] for r in model_results]
        all_r2 = [r['r2'] for r in model_results]

        validation_stats[model_name] = {
            'rmse_mean': np.mean(all_rmse),
            'rmse_std': np.std(all_rmse),
            'r2_mean': np.mean(all_r2),
            'r2_std': np.std(all_r2),
            'n_windows': len(windows),
            'n_scores': len(model_results),
            'window_stats': window_stats
        }

    return validation_stats

def calculate_test_statistics(final_results):
    """Calculate statistics for final test set results (last year)."""
    test_stats = {}

    # Group results by model
    model_results = {}
    for result in final_results:
        model_name = result['model'].split('_')[0].lower()  # Extract base model name
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)

    # Calculate statistics for each model
    for model_name, results in model_results.items():
        if not results:
            continue

        rmse_scores = [r['rmse'] for r in results]
        r2_scores = [r['r2'] for r in results]

        test_stats[model_name] = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores) if len(rmse_scores) > 1 else 0.0,
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores) if len(r2_scores) > 1 else 0.0,
            'n_scores': len(results)
        }

    return test_stats

def main():
    # Load results
    print("Loading model results...")
    results_data = load_results()

    # Calculate validation statistics
    print("\nCalculating validation statistics...")
    validation_stats = calculate_validation_statistics(results_data['validation_results'])

    # Calculate test statistics
    print("\nCalculating final test set statistics...")
    test_stats = calculate_test_statistics(results_data['final_results'])

    # Print validation results
    print("\n=== Validation Results (Cross-window Statistics) ===")
    baseline_stats = None
    for model_name, stats in validation_stats.items():
        print(f"\n{model_name.upper()}")
        print(f"Number of windows: {stats['n_windows']}")
        print(f"Total scores: {stats['n_scores']}")
        print(f"RMSE - Mean: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}")
        print(f"R² - Mean: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")

        if model_name == 'baseline':
            baseline_stats = stats
        elif baseline_stats is not None:
            rmse_diff = baseline_stats['rmse_mean'] - stats['rmse_mean']
            rmse_std = np.sqrt(baseline_stats['rmse_std']**2 + stats['rmse_std']**2)
            std_above_baseline = rmse_diff / rmse_std if rmse_std > 0 else float('inf')
            print(f"Standard deviations above baseline (RMSE): {std_above_baseline:.2f}")

    # Print test results
    print("\n=== Final Test Set Results (Last Year) ===")
    baseline_test_stats = None
    for model_name, stats in test_stats.items():
        print(f"\n{model_name.upper()}")
        print(f"Number of evaluations: {stats['n_scores']}")
        print(f"RMSE - Mean: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}")
        print(f"R² - Mean: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")

        if model_name == 'baseline':
            baseline_test_stats = stats
        elif baseline_test_stats is not None:
            rmse_diff = baseline_test_stats['rmse_mean'] - stats['rmse_mean']
            rmse_std = np.sqrt(baseline_test_stats['rmse_std']**2 + stats['rmse_std']**2)
            std_above_baseline = rmse_diff / rmse_std if rmse_std > 0 else float('inf')
            print(f"Standard deviations above baseline (RMSE): {std_above_baseline:.2f}")

    # Create summary DataFrame for validation results
    validation_summary = {
        model: {
            'Validation RMSE (mean ± std)': f"{stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}",
            'Validation R² (mean ± std)': f"{stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}",
            'Validation Windows': stats['n_windows'],
            'Validation Scores': stats['n_scores']
        }
        for model, stats in validation_stats.items()
    }

    # Add test results to summary
    for model, stats in test_stats.items():
        if model in validation_summary:
            validation_summary[model].update({
                'Test RMSE (mean ± std)': f"{stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}",
                'Test R² (mean ± std)': f"{stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}",
                'Test Scores': stats['n_scores']
            })

    # Save summary
    summary_df = pd.DataFrame(validation_summary).T
    summary_df.to_csv('model_statistics_summary.csv')

    # Save detailed statistics
    with open('detailed_statistics.pkl', 'wb') as f:
        pickle.dump({
            'validation_stats': validation_stats,
            'test_stats': test_stats
        }, f)

    print("\nSummary saved to 'model_statistics_summary.csv'")
    print("Detailed statistics saved to 'detailed_statistics.pkl'")

if __name__ == "__main__":
    main()
