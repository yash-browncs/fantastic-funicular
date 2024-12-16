import pickle
import pandas as pd
import numpy as np

def load_results():
    with open('model_results.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_parameters(results):
    best_params = results['best_params']
    final_results = results['final_results']

    # Create summary dataframe
    summary = []
    for model_name, params in best_params.items():
        # Get performance metrics
        model_results = [r for r in final_results if model_name in r['model']]
        if model_results:
            rmse_scores = [r['rmse'] for r in model_results]
            r2_scores = [r['r2'] for r in model_results]

            summary.append({
                'Model': model_name,
                'Best Parameters': str(params),
                'Mean RMSE': f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}",
                'Mean R²': f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}"
            })

    return pd.DataFrame(summary)

def main():
    print("Loading results...")
    results = load_results()

    print("\n=== Best Parameters and Performance Summary ===")
    summary_df = analyze_parameters(results)
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_df.to_csv('model_parameters_summary.csv', index=False)
    print("\nSummary saved to model_parameters_summary.csv")

if __name__ == "__main__":
    main()
