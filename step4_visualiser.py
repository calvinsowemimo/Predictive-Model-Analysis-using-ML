import pandas as pd

# Adjust the file path as necessary
file_path = 'path\\New Cleaned Data\\ftse100_merged_data.csv'
ftse100_df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format for time series analysis
ftse100_df['Date'] = pd.to_datetime(ftse100_df['Date'])
ftse100_df.set_index('Date', inplace=True)  # Setting the Date as index for easier time series analysis

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = ftse100_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of FTSE 100 and Economic Indicators')
plt.show()

#Time Series Plot
plt.figure(figsize=(14, 7))
plt.plot(ftse100_df.index, ftse100_df['Close'], label='Close Price')
plt.plot(ftse100_df.index, ftse100_df['Moving_Average_7'], label='7-Day Moving Average', linestyle='--')
plt.plot(ftse100_df.index, ftse100_df['Moving_Average_30'], label='30-Day Moving Average', linestyle='--')
plt.title('FTSE 100 Close Price and Moving Averages')
plt.legend()
plt.show()

# Creating a scatter plot function for reusability
def plot_scatter(x, y, x_label, y_label='Close Price', data=ftse100_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x], data[y], alpha=0.5)
    plt.title(f'{x_label} vs. {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# GDP vs. Close Price
plot_scatter('GDP', 'Close', 'GDP')

# Inflation Rate vs. Close Price
plot_scatter('Inflation Rate', 'Close', 'Inflation Rate')

# Bank Rate vs. Close Price
plot_scatter('Bank Rate', 'Close', 'Bank Rate')

# Unemployment Rate vs. Close Price
plot_scatter('Unemploment Rate', 'Close', 'Unemployment Rate')

# Sentiment vs. Close Price
plot_scatter('Sentiment', 'Close', 'Sentiment')
