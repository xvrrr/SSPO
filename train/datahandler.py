import os
import shutil
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


class StockDataDownloader:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.combined_data = None
        self.close_prices = None

    def download_data(self):
        # Initialize an empty list to store DataFrames
        data_frames = []

        # Download data for each symbol and append to the list
        for symbol in self.symbols:
            data = yf.download(symbol, start=self.start_date, end=self.end_date)
            data['Symbol'] = symbol  # Add a column for the ticker symbol
            data_frames.append(data)

        # Concatenate all DataFrames vertically
        self.combined_data = pd.concat(data_frames)

        # Reset index for a clean DataFrame
        self.combined_data.reset_index(inplace=True)
        self.combined_data.set_index('Date', inplace=True)

    def preprocess_data(self):
        # Get unique symbols
        symbols_list = self.combined_data['Symbol'].unique().tolist()
        selected_column = ['Adj Close']

        # Create a DataFrame for close prices
        daterange = self.combined_data.index.unique().sort_values()
        self.close_prices = pd.DataFrame(index=daterange)

        for symbol in symbols_list:
            df_sym = self.combined_data[self.combined_data['Symbol'] == symbol]
            df_tmp = df_sym[selected_column]
            df_tmp.columns = [symbol]
            self.close_prices = pd.concat([self.close_prices, df_tmp], axis=1)

    def save_to_csv(self, filename):
        if self.close_prices is not None:
            self.close_prices.to_csv(filename)
        else:
            print("Close prices DataFrame is empty. Please run preprocess_data() first.")

    def run(self, filename):
        self.download_data()
        self.preprocess_data()
        self.save_to_csv(filename)


def process_data(file_name, symbols, TOPERCENTAGE=100):
    # Read the CSV file
    close_prices_org = pd.read_csv(file_name, index_col=0, parse_dates=True)
    close_prices = close_prices_org[symbols]
    names = close_prices.columns.to_list()

    # Clean the data by removing NaN values
    close_prices = close_prices.dropna(how='all')
    close_prices.fillna(method='ffill', inplace=True)
    close_prices.fillna(method='bfill', inplace=True)

    close_prices_assets=close_prices

    # Calculate log returns
    returns = pd.DataFrame({})
    for name in names:
        current_returns = np.log(close_prices[name] / close_prices[name].shift(1))
        returns[name] = current_returns.iloc[1:] * TOPERCENTAGE  # Remove NaN from first row

    # Store returns in returns_p (as a copy of returns)
    returns_p = returns.copy()

    # Calculate mean return and covariance matrix
    mean_return = returns.mean()
    cov = returns.cov()
    cov_np = cov.to_numpy()

    # Calculate pct change and normalize
    pct_change = close_prices.pct_change().values[1:]
    normalized_pct_change = pct_change + 1

    assets_value = close_prices.values[1:]

    # Concatenate data: asset values and normalized pct change
    data_w_ret = np.concatenate([assets_value, normalized_pct_change], axis=1)

    # Convert to tensor
    data_tensor = torch.tensor(data_w_ret, dtype=torch.float32).unsqueeze(0)

    # Print the shapes and values for debugging
    print(f"Processed data tensor shape: {data_tensor.shape}")
    print(f"Mean return: {mean_return}")
    print(f"Covariance matrix: \n{cov}")
    print(f"Covariance matrix (numpy): \n{cov_np}")
    
    return data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names



if __name__ == "__main__":

    symbols = ['AAPL', 'AMGN', 'AMZN', 'BRK-B', 'C', 'COST', 'DHR', 'GE',  'GOOGL',  'IBM',  'JNJ',  'KO',  'LLY',  'MA',  'MCD',  'META',  'MSFT',  'NVDA',  'PEP',  'PG',  'RTX',  'SO',  'TSLA', 'UNH','USB','WFC', 'WMT', 'AMD','DIS','NFLX']
    




    downloader = StockDataDownloader(symbols, "2023-06-13", "2024-06-11")
    downloader.run('index.csv')

   
    
