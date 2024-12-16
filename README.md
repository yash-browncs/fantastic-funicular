# Szeged Weather Temperature Prediction

This repository contains code for analyzing and predicting next-hour temperatures using the Szeged Weather dataset.

## Dataset
The dataset contains hourly weather measurements from Szeged, Hungary (2005-2016) including:
- Temperature
- Humidity
- Wind speed
- Wind bearing
- Visibility
- Pressure
- Precipitation type
- Weather summary

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract the dataset:
```bash
unzip weather_data.zip
```

## Running the Analysis

1. Run EDA:
```bash
python eda_script.py
```

2. Preprocess data:
```bash
python preprocess_data.py
```

3. Train models:
```bash
python train_models.py
```

4. Analyze results:
```bash
python plot_test_scores.py
```

## Files
- eda_script.py: Exploratory Data Analysis
- preprocess_data.py: Data preprocessing
- split_data.py: Data splitting utilities
- train_models.py: Model training
- model_utils.py: Model classes and utilities
- evaluate_models.py: Model evaluation
- plot_test_scores.py: Visualization of results
- feature_importance.py: Feature importance analysis
- analyze_shap.py: SHAP value analysis
