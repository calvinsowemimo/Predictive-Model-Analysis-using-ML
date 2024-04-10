# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import re
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

# Function to load data
def load_data(file_paths):
    data_frames = {}
    for key, info in file_paths.items():
        path = info['path']
        data_frames[key] = pd.read_csv(path, low_memory=False)
    return data_frames

# Function to clean data
def clean_data(data_frames):
    for name, df in data_frames.items():
        df.drop_duplicates(inplace=True)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            
    return data_frames

# Function to add sentiment score
def add_sentiment_score(data_frames, text_columns):
    for name, column in text_columns.items():
        df = data_frames[name]
        if column in df.columns:
            df['Sentiment'] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        else:
            print(f"Column '{column}' not found in DataFrame '{name}'. Skipping sentiment analysis.")
    return data_frames

# Function to add moving averages
def add_moving_averages(data_frames, window_sizes=[7, 30]):
    for name, df in data_frames.items():
        if 'Adj Close' in df.columns:
            # Ensure 'Adj Close' is treated as numeric, coercing any errors
            df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
            # df['Adj Close'] = df['Adj Close'].fillna(df['Adj Close'].mean())
            
            for window in window_sizes:
                moving_avg_col_name = f'Moving_Average_{window}'
                df[moving_avg_col_name] = df['Adj Close'].rolling(window=window).mean()
    return data_frames

# Function to normalize numerical data
def normalize_data(data_frames, columns_to_normalize):
    scaler = StandardScaler()
    for name, columns in columns_to_normalize.items():
        df = data_frames[name]
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        else:
            print(f"No numeric columns found for normalization in DataFrame '{name}'. Skipping.")
    return data_frames

# Function to save cleaned data
def save_cleaned_data(data_frames, save_paths):
    for name, path in save_paths.items():
        df = data_frames[name]
        df.to_csv(path, index=False)
    print("Data saved successfully.")

# Main function to orchestrate data preprocessing
def main():
    file_paths = {
        'ftse100_index_5y': {'path': 'path\\ftse100_index_5y.csv'},
        'tweets': {'path': 'path\\ftse100_tweets.csv'},
        'news': {'path': 'path\\news_data.csv'},
        'gdp': {'path': 'path\\uk_gdp_growth.csv'},
        'inflation': {'path': 'path\\uk_inflation_rates.csv'},
        'interest': {'path': 'path\\uk_interest_rates.csv'},
        'unemployment': {'path': 'path\\uk_unemployment_rates.csv'}
    }

    data_frames = load_data(file_paths)
    data_frames = clean_data(data_frames)
    data_frames = add_moving_averages(data_frames)
    text_columns = {'tweets': 'Tweet', 'news': 'Headline'}
    data_frames = add_sentiment_score(data_frames, text_columns)
    
    columns_to_normalize = {
        'ftse100_index_5y': ['Adj Close', 'Moving_Average_7', 'Moving_Average_30']
    }
    data_frames = normalize_data(data_frames, columns_to_normalize)
    
    save_paths = {
        'ftse100_index_5y': 'path\\ftse100_index_5y_clean.csv',
        'tweets': 'path\\ftse100_tweets_clean.csv',
        'news': 'path\\news_data_clean.csv',
        'gdp': 'path\\uk_gdp_growth_clean.csv',
        'inflation': 'path\\uk_inflation_rates_clean.csv',
        'interest': 'path\\uk_interest_rates_clean.csv',
        'unemployment': 'path\\uk_unemployment_rates_clean.csv'
    }

    save_cleaned_data(data_frames, save_paths)

if __name__ == '__main__':
    main()