import pandas as pd

# Load data
df = pd.read_csv('weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# Print dataset information
print('\nDataset Date Range:')
print(f'Start: {df["Formatted Date"].min()}')
print(f'End: {df["Formatted Date"].max()}')
print(f'Total duration: {df["Formatted Date"].max() - df["Formatted Date"].min()}')
print(f'Total records: {len(df)}')

# Calculate last year
last_year_start = df['Formatted Date'].max() - pd.Timedelta(days=365)
train_start = df['Formatted Date'].min()

print('\nProposed Split Points:')
print(f'Training/Validation Period: {train_start} to {last_year_start}')
print(f'Test Period (Last Year): {last_year_start} to {df["Formatted Date"].max()}')

# Calculate sample sizes
train_val_samples = len(df[df['Formatted Date'] < last_year_start])
test_samples = len(df[df['Formatted Date'] >= last_year_start])

print('\nSample Sizes:')
print(f'Training/Validation samples: {train_val_samples}')
print(f'Test samples: {test_samples}')

# Calculate potential window sizes
train_val_duration = last_year_start - train_start
window_size = pd.Timedelta(days=365.25 * 5.5)  # 5 years training + 6 months validation
shift_size = pd.Timedelta(days=365.25/2)  # 6 months shift

n_windows = 0
current_start = train_start
while current_start + window_size <= last_year_start:
    n_windows += 1
    current_start += shift_size

print(f'\nPotential Window Information:')
print(f'Training/Validation Period Duration: {train_val_duration}')
print(f'Window Size (5.5 years): {window_size}')
print(f'Window Shift (6 months): {shift_size}')
print(f'Number of possible windows: {n_windows}')
