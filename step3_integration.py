### -- ECONOMIC INDICATOR INTEGRATION --
import pandas as pd

# Read
ftse100_df = pd.read_csv('path\\ftse100_index_5y_clean.csv')
gdp_growth_df = pd.read_csv('path\\uk_gdp_growth_clean.csv')
inflation_rates_df = pd.read_csv('path\\uk_inflation_rates_clean.csv')
interest_rates_df = pd.read_csv('path\\uk_interest_rates_clean.csv')
unemployment_rates_df = pd.read_csv('path\\uk_unemployment_rates_clean.csv')

# Correctly convert 'Date' columns to datetime in all DataFrames
ftse100_df['Date'] = pd.to_datetime(ftse100_df['Date'])
gdp_growth_df['Date'] = pd.to_datetime(gdp_growth_df['Date'])
inflation_rates_df['Date'] = pd.to_datetime(inflation_rates_df['Date'])
interest_rates_df['Date'] = pd.to_datetime(interest_rates_df['Date'])
unemployment_rates_df['Date'] = pd.to_datetime(unemployment_rates_df['Date'])

# Merging DataFrames on 'Date' column
ftse100_df = ftse100_df.merge(gdp_growth_df, on='Date', how='left')
ftse100_df = ftse100_df.merge(inflation_rates_df, on='Date', how='left')
ftse100_df = ftse100_df.merge(interest_rates_df, on='Date', how='left')
ftse100_df = ftse100_df.merge(unemployment_rates_df, on='Date', how='left')

# Backfilling NAN
ftse100_df[['GDP', 'Inflation Rate', 'Unemploment Rate', 'Moving_Average_7', 'Moving_Average_30']] = ftse100_df[['GDP', 'Inflation Rate', 'Unemploment Rate', 'Moving_Average_7', 'Moving_Average_30']].fillna(method='bfill')

# Check the first few rows to confirm successful merge
print(ftse100_df.head())

# Check for any NaN values that may need addressing
print(ftse100_df.isnull().sum())

### -- SENTIMENTAL INTEGRATION --
import pandas as pd

# Replace the path with the actual path to your news data CSV file
news_df = pd.read_csv('path\\news_data_clean.csv')

# Ensure the 'Date' column is in datetime format
news_df['Date'] = pd.to_datetime(news_df['Date'])

# Quick check of the data
print(news_df.head())

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get the compound sentiment score
def get_sentiment_score(text):
    return sid.polarity_scores(text)['compound']

# Apply the function to your headlines or news content
news_df['Sentiment'] = news_df['Headline'].apply(get_sentiment_score)

# Aggregate sentiment scores by date if you have multiple articles per day
daily_sentiment = news_df.groupby('Date')['Sentiment'].mean().reset_index()

# Assuming your FTSE 100 DataFrame is named ftse100_df and it already has a 'Date' column in datetime format
ftse100_df = ftse100_df.merge(daily_sentiment, on='Date', how='left')

# Fill any missing sentiment scores with 0 or another method of your choice
ftse100_df['Sentiment'] = ftse100_df['Sentiment'].fillna(0)

# Check the merged DataFrame
print(ftse100_df.head())

# Save the merged DataFrame to a new CSV file
ftse100_df.to_csv('path\\ftse100_merged_data.csv', index=False)
print("Data Has been Merged and Succesfully Saved to a CSV")