import yfinance as yf

# Define the ticker symbol for the FTSE 100 Index.
ftse100_symbol = "^FTSE"

# Start and end dates for the period.
start_date = "2018-01-01"
end_date = "2023-01-01"

#yfinance to download the data
ftse100_data = yf.download(ftse100_symbol, start=start_date, end=end_date)

# Check the first few rows to ensure data was fetched correctly
print(ftse100_data.head())

# Save the data to a CSV file
ftse100_data.to_csv('path\\ftse100_index_5y.csv')
