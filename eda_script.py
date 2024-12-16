import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set basic style parameters
plt.rcParams['figure.figsize'] = [20, 25]
plt.rcParams['figure.dpi'] = 100

# Load and prepare data
df = pd.read_csv('weatherHistory.csv')
# Parse datetime with explicit UTC=True to handle timezone
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
# Extract components after parsing
df['Year'] = df['Formatted Date'].dt.year
df['Month'] = df['Formatted Date'].dt.month
df['Hour'] = df['Formatted Date'].dt.hour

# Create figure with subplots
fig = plt.figure()

# 1. Temperature Distribution (Histogram)
plt.subplot(4, 2, 1)
plt.hist(df['Temperature (C)'], bins=50, color='skyblue', edgecolor='black')
plt.title('Temperature Distribution', fontsize=12)
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')

# 2. Temperature vs Humidity Scatter Plot
plt.subplot(4, 2, 2)
scatter = plt.scatter(df['Humidity'], df['Temperature (C)'],
                     c=df['Wind Speed (km/h)'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Wind Speed (km/h)')
plt.title('Temperature vs Humidity (color = Wind Speed)', fontsize=12)
plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')

# 3. Monthly Temperature Box Plot
plt.subplot(4, 2, 3)
df.boxplot(column='Temperature (C)', by='Month', ax=plt.gca())
plt.title('Temperature Distribution by Month', fontsize=12)
plt.xlabel('Month')
plt.ylabel('Temperature (C)')

# 4. Hourly Temperature Box Plot
plt.subplot(4, 2, 4)
df.boxplot(column='Temperature (C)', by='Hour', ax=plt.gca())
plt.title('Temperature Distribution by Hour', fontsize=12)
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (C)')

# 5. Correlation Heatmap
plt.subplot(4, 2, 5)
numeric_cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                'Pressure (millibars)']
correlation = df[numeric_cols].corr()
im = plt.imshow(correlation, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Feature Correlations', fontsize=12)

# 6. Temperature by Precipitation Type (Box Plot)
plt.subplot(4, 2, 6)
df.boxplot(column='Temperature (C)', by='Precip Type', ax=plt.gca())
plt.title('Temperature Distribution by Precipitation Type', fontsize=12)
plt.xlabel('Precipitation Type')
plt.ylabel('Temperature (C)')

# 7. Yearly Temperature Trends (Line Plot)
plt.subplot(4, 2, 7)
yearly_temp = df.groupby('Year')['Temperature (C)'].mean()
plt.plot(yearly_temp.index, yearly_temp.values, marker='o', color='blue')
plt.title('Average Temperature by Year', fontsize=12)
plt.xlabel('Year')
plt.ylabel('Average Temperature (C)')

# 8. Wind Speed vs Temperature (Scatter Plot)
plt.subplot(4, 2, 8)
plt.scatter(df['Wind Speed (km/h)'], df['Temperature (C)'], alpha=0.5, color='green')
plt.title('Temperature vs Wind Speed', fontsize=12)
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Temperature (C)')

plt.tight_layout()
plt.savefig('weather_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# Calculate and print key statistics
print("\n=== Key Insights ===")

print("\n1. Temperature Statistics by Season:")
df['Season'] = pd.cut(df['Month'],
                     bins=[0,3,6,9,12],
                     labels=['Winter', 'Spring', 'Summer', 'Fall'])
season_stats = df.groupby('Season')['Temperature (C)'].describe()
print(season_stats)

print("\n2. Temperature Range:")
print(f"Maximum Temperature: {df['Temperature (C)'].max():.2f}°C")
print(f"Minimum Temperature: {df['Temperature (C)'].min():.2f}°C")
print(f"Average Temperature: {df['Temperature (C)'].mean():.2f}°C")

print("\n3. Correlation with Temperature:")
correlations = df[numeric_cols].corr()['Temperature (C)'].sort_values(ascending=False)
print(correlations)

print("\n4. Hourly Temperature Patterns:")
hourly_stats = df.groupby('Hour')['Temperature (C)'].agg(['mean', 'std']).round(2)
print(hourly_stats)

print("\n5. Precipitation Type Distribution:")
print(df['Precip Type'].value_counts(dropna=False))
