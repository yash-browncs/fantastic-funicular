import pandas as pd
from train_models import create_walk_forward_windows

def print_detailed_window_info(windows, df):
    """Print detailed information about the windows."""
    total_span = df['Formatted Date'].max() - df['Formatted Date'].min()
    print(f'\nTotal dataset span: {total_span.days/365.25:.2f} years')
    print(f'Date range: {df["Formatted Date"].min().strftime("%Y-%m-%d")} to {df["Formatted Date"].max().strftime("%Y-%m-%d")}')

    print(f'\nNumber of windows created: {len(windows)}')

    print('\nWindow details:')
    for i, window in enumerate(windows, 1):
        print(f'\nWindow {i}:')
        for split, (start, end) in window.items():
            duration = end - start
            samples = len(df[(df['Formatted Date'] >= start) & (df['Formatted Date'] < end)])
            print(f'{split}:')
            print(f'  Period: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}')
            print(f'  Duration: {duration.days/365.25:.2f} years')
            print(f'  Samples: {samples:,}')

        if i == 1 or i == len(windows):
            print(f'{"First" if i == 1 else "Last"} window verification:')
            train_duration = window['train'][1] - window['train'][0]
            val_duration = window['val'][1] - window['val'][0]
            test_duration = window['test'][1] - window['test'][0]
            print(f'  Training duration: {train_duration.days/365.25:.2f} years (expected: 5.0)')
            print(f'  Validation duration: {val_duration.days/365.25:.2f} years (expected: 0.5)')
            print(f'  Test duration: {test_duration.days/365.25:.2f} years (expected: 0.5)')

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('weatherHistory.csv')
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

    # Create windows
    print("\nCreating windows...")
    windows = create_walk_forward_windows(df)

    # Print detailed window information
    print_detailed_window_info(windows, df)

if __name__ == "__main__":
    main()
