import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_results():
    with open('model_results.pkl', 'rb') as f:
        return pickle.load(f)

def extract_model_scores(results, model_name):
    # Extract scores for the specified model
    scores = []
    for result in results['final_results']:
        if model_name in result['model']:
            scores.append({
                'rmse': result['rmse'],
                'r2': result['r2'],
                'random_state': result.get('random_state', None)
            })
    return scores

def calculate_statistics(scores):
    rmse_values = [s['rmse'] for s in scores]
    r2_values = [s['r2'] for s in scores]

    return {
        'rmse': {
            'mean': np.mean(rmse_values),
            'std': np.std(rmse_values)
        },
        'r2': {
            'mean': np.mean(r2_values),
            'std': np.std(r2_values)
        }
    }

def plot_scores_with_error_bars():
    # Set style
    plt.style.use('seaborn-v0_8')

    # Load results
    results = load_results()

    # Extract scores for XGBoost and Random Forest
    xgb_scores = extract_model_scores(results, 'xgb')
    rf_scores = extract_model_scores(results, 'rf')

    # Calculate statistics
    xgb_stats = calculate_statistics(xgb_scores)
    rf_stats = calculate_statistics(rf_scores)

    # Prepare data for plotting
    models = ['XGBoost', 'Random Forest']
    rmse_means = [xgb_stats['rmse']['mean'], rf_stats['rmse']['mean']]
    rmse_stds = [xgb_stats['rmse']['std'], rf_stats['rmse']['std']]
    r2_means = [xgb_stats['r2']['mean'], rf_stats['r2']['mean']]
    r2_stds = [xgb_stats['r2']['std'], rf_stats['r2']['std']]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RMSE
    ax1.bar(models, rmse_means, yerr=rmse_stds, capsize=5)
    ax1.set_title('RMSE Scores with Error Bars')
    ax1.set_ylabel('RMSE')

    # Plot R²
    ax2.bar(models, r2_means, yerr=r2_stds, capsize=5)
    ax2.set_title('R² Scores with Error Bars')
    ax2.set_ylabel('R²')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('test_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\nModel Performance Statistics:")
    print("\nXGBoost:")
    print(f"RMSE: {xgb_stats['rmse']['mean']:.4f} ± {xgb_stats['rmse']['std']:.4f}")
    print(f"R²: {xgb_stats['r2']['mean']:.4f} ± {xgb_stats['r2']['std']:.4f}")
    print("\nRandom Forest:")
    print(f"RMSE: {rf_stats['rmse']['mean']:.4f} ± {rf_stats['rmse']['std']:.4f}")
    print(f"R²: {rf_stats['r2']['mean']:.4f} ± {rf_stats['r2']['std']:.4f}")

if __name__ == "__main__":
    plot_scores_with_error_bars()
