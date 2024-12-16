import pandas as pd
from train_models import create_walk_forward_windows, verify_windows

# Load data
df = pd.read_csv('weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# Create and verify windows
windows, last_year_start = create_walk_forward_windows(df)
verify_windows(windows)

# Print window sizes
if len(windows) > 0:
    first_window = windows[0]
    train_size = len(df[(df['Formatted Date'] >= first_window['train'][0]) &
                       (df['Formatted Date'] < first_window['train'][1])])
    val_size = len(df[(df['Formatted Date'] >= first_window['val'][0]) &
                      (df['Formatted Date'] < first_window['val'][1])])

    print(f'\nSample sizes for first window:')
    print(f'Training samples: {train_size}')
    print(f'Validation samples: {val_size}')
    print(f'Last year start: {last_year_start}')

    # Print date ranges
    print(f'\nDate ranges for first window:')
    print(f'Training: {first_window["train"][0]} to {first_window["train"][1]}')
    print(f'Validation: {first_window["val"][0]} to {first_window["val"][1]}')
