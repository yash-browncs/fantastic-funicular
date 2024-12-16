import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import shap
from model_utils import XGBRegressorWrapper

def load_data_and_models():
    try:
        with open('model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        with open('data_splits.pkl', 'rb') as f:
            data = pickle.load(f)
        return results, data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def calculate_permutation_importance(model, X_test, y_test, n_repeats=5):
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
        return pd.DataFrame(
            {'importance': r.importances_mean, 'std': r.importances_std},
            index=X_test.columns
        ).sort_values('importance', ascending=False)
    except Exception as e:
        print(f"Error calculating permutation importance: {str(e)}")
        return pd.DataFrame()

def calculate_linear_importance(model, X):
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        importance = pd.DataFrame(
            {'importance': np.abs(model.coef_)},
            index=X.columns
        ).sort_values('importance', ascending=False)
        return importance
    except Exception as e:
        print(f"Error calculating linear importance: {str(e)}")
        return pd.DataFrame()

def calculate_xgboost_importance(model):
    try:
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        importances = {}
        xgb_model = model.model if isinstance(model, XGBRegressorWrapper) else model
        for imp_type in importance_types:
            importance = xgb_model.get_booster().get_score(importance_type=imp_type)
            importances[imp_type] = pd.Series(importance).sort_values(ascending=False)
        return importances
    except Exception as e:
        print(f"Error calculating XGBoost importance: {str(e)}")
        return {}

def calculate_shap_values(model, X):
    try:
        if isinstance(model, XGBRegressorWrapper):
            model = model.model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return {
            'values': shap_values,
            'expected_value': explainer.expected_value
        }
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        return None

def summarize_feature_importance(importances):
    summary = {}

    for name in ['ridge_permutation', 'lasso_permutation', 'rf_permutation', 'xgb_permutation']:
        if name in importances:
            summary[f'Top 5 features ({name})'] = importances[name].head()

    for name in ['ridge_coefficients', 'lasso_coefficients']:
        if name in importances:
            summary[f'Top 5 features ({name})'] = importances[name].head()

    if 'xgboost_metrics' in importances:
        for metric, values in importances['xgboost_metrics'].items():
            summary[f'Top 5 features (XGBoost {metric})'] = values.head()

    return summary

def main():
    print("Loading data and models...")
    results, data = load_data_and_models()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    importances = {}

    print("\nCalculating permutation importance for selected models...")
    for name, model in results['models'].items():
        if name not in ['baseline', 'svr']:
            print(f"Processing {name}...")
            importances[f'{name}_permutation'] = calculate_permutation_importance(
                model, X_test, y_test, n_repeats=5
            )

    print("\nCalculating linear model importance...")
    for name in ['ridge', 'lasso']:
        print(f"Processing {name}...")
        importances[f'{name}_coefficients'] = calculate_linear_importance(
            results['models'][name], X_train
        )

    print("\nCalculating XGBoost importance metrics...")
    importances['xgboost_metrics'] = calculate_xgboost_importance(
        results['models']['xgb']
    )

    print("\nCalculating SHAP values...")
    shap_values = {}
    for name in ['rf', 'xgb']:
        print(f"Processing {name}...")
        shap_values[name] = calculate_shap_values(
            results['models'][name], X_test.iloc[:1000]
        )

    print("\nGenerating feature importance summary...")
    importance_summary = summarize_feature_importance(importances)

    print("\nSaving feature importance and SHAP results...")
    with open('feature_importance.pkl', 'wb') as f:
        pickle.dump({
            'importances': importances,
            'importance_summary': importance_summary,
            'shap_values': shap_values,
            'test_data': X_test
        }, f)

    print("\nFeature Importance Summary:")
    for name, summary in importance_summary.items():
        print(f"\n{name}:")
        print(summary)

    print("\nFeature importance analysis complete!")

if __name__ == "__main__":
    main()
