import yfinance as yf
from datetime import datetime, timedelta
from temporary_data_storage import TemporaryDataStorage
import pandas as pd

# Define the stock tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'HYG', 'TLT', 'LQD', 'XLK', 'XLY', 'XLI', 'XLF', 'XLE',
           'XLB', 'IYR', 'XLV', 'XLP', 'XLU', 'HG=F', '^RUT', '^IXIC', '^DJI', 'DX-Y.NYB', 'SHY',
           'IEF', 'IEI', 'SPY', 'NVDA', 'SMH', 'AMZN', 'XHB', 'GLD', '^VIX'] # Add or remove tickers as needed


# Define the time period (last 7 years from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=6100)).strftime('%Y-%m-%d')

# Initialize a dictionary to store the closing prices
closing_prices = {ticker: [] for ticker in tickers}
opening_prices = {ticker: [] for ticker in tickers}
high_prices = {ticker: [] for ticker in tickers}
low_prices = {ticker: [] for ticker in tickers}
volumes = {ticker: [] for ticker in tickers}



# Initialize a dictionary to store the data
data_frames = {ticker: pd.DataFrame() for ticker in tickers}

# Fetch the historical market data
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data_frames[ticker] = data

# Find the maximum start date and minimum end date across all tickers
max_start = max(df.index.min() for df in data_frames.values() if not df.empty)
min_end = min(df.index.max() for df in data_frames.values() if not df.empty)

# Truncate the data frames to the common date range
for ticker, df in data_frames.items():
    if not df.empty:
        data_frames[ticker] = df.loc[max_start:min_end]

# Now extract the aligned data
for ticker in tickers:
    df = data_frames[ticker]
    if not df.empty:
        closing_prices[ticker] = df['Close'].tolist()
        opening_prices[ticker] = df['Open'].tolist()
        high_prices[ticker] = df['High'].tolist()
        low_prices[ticker] = df['Low'].tolist()
        volumes[ticker] = df['Volume'].tolist()

market_data = {
    'closing_prices': closing_prices,
    'opening_prices': opening_prices,
    'high_prices': high_prices,
    'low_prices': low_prices,
    'volumes': volumes,
}

storage = TemporaryDataStorage("market_data.pkl")
storage.save_data(market_data)

# Now, closing_prices dictionary contains the historical closing prices for each ticker
