import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

def load_results():
    """Load feature importance results including SHAP values"""
    with open('feature_importance.pkl', 'rb') as f:
        return pickle.load(f)

def create_shap_summary_plot(shap_values, features, model_name):
    """Create and save SHAP summary plot"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values['values'], features, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{model_name}.png')
    plt.close()

def create_shap_dependence_plots(shap_values, features, model_name):
    """Create dependence plots for top features"""
    # Get feature names sorted by mean absolute SHAP value
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': np.abs(shap_values['values']).mean(0)
    }).sort_values('importance', ascending=False)

    # Create dependence plots for top 3 features
    for feature in feature_importance['feature'][:3]:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_values['values'],
            features,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature} ({model_name})')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{model_name}_{feature.replace(" ", "_")}.png')
        plt.close()

def analyze_shap_values():
    """Analyze SHAP values and create visualizations"""
    # Load results
    results = load_results()
    shap_values = results['shap_values']
    test_data = results['test_data'].iloc[:1000]  # Match the subset used for SHAP calculation

    # Create visualizations for each model
    for model_name, model_shap in shap_values.items():
        if model_shap is not None:
            print(f"\nAnalyzing SHAP values for {model_name}...")
            create_shap_summary_plot(model_shap, test_data, model_name)
            create_shap_dependence_plots(model_shap, test_data, model_name)

            # Calculate and print mean absolute SHAP values
            mean_abs_shap = pd.DataFrame({
                'feature': test_data.columns,
                'mean_abs_shap': np.abs(model_shap['values']).mean(0)
            }).sort_values('mean_abs_shap', ascending=False)

            print(f"\nTop 10 most important features for {model_name} (by mean |SHAP|):")
            print(mean_abs_shap.head(10))

            print(f"\nLeast important features for {model_name} (by mean |SHAP|):")
            print(mean_abs_shap.tail(5))

if __name__ == "__main__":
    analyze_shap_values()
