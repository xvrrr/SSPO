import torch
import torch.nn as nn
import sys
sys.path.append("..")
from model.PO_DL import *
import argparse
import yfinance as yf
import pandas as pd
import numpy as np

def train_model(model, train_data, epochs=850, learning_rate=0.001, batch_size=64, log_dir='runs/experiment0'):
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    # Creating PyTorch datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    progress_bar = tqdm(total=epochs, desc='Training Progress', unit='epoch', position=0, ncols=200)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            batch_data = batch[0]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            allocations = model(batch_data)
            
            # Compute loss
            loss = model.sharpe_loss(allocations, batch_data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    

        # Normalize losses by number of batches
        train_loss /= len(train_dataloader)

        # Log losses
        writer.add_scalar('Loss/train', train_loss, epoch)




        if epoch % 1 == 0:
            progress_bar.set_postfix({
                'Train Loss': train_loss,
            })
            progress_bar.update(1)  

    progress_bar.close()
        
    writer.close()












def main():
    # Define symbols (example)
    symbols = ['AAPL',   'AMGN','AMZN',    'BRK-B',   'C',   'COST', 'DHR', 'GE',  'GOOGL',  'IBM',  'JNJ',  'KO',  'LLY',  'MA',  'MCD',  'META',  'MSFT',  'NVDA',  'PEP',  'PG',  'RTX',  'SO',  'TSLA', 'UNH','USB','WFC', 'WMT', 'AMD','DIS','NFLX']
    
    # Process and save train data
    data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names = process_data('index.csv', symbols)


 

    return data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names




def download_stock_data(Market_index, start, end):

    # Download data for each symbol and store in a list
    data = []
    for symbol in Market_index:
        stock_data = yf.download(symbol, start=start, end=end)['Adj Close']
        data.append(stock_data)


    # Merge all data into a single DataFrame
    merged_df = pd.concat(data, axis=1, join='outer')
    merged_df.columns = Market_index  # Rename columns with the stock symbols


    # Calculate log returns
    orig_merged_df = merged_df
    orig_merged_df['Index'] = range(len(orig_merged_df))
    orig_merged_df.set_index('Index', inplace=True)


    return orig_merged_df









def DCfilter(data, d=0.07):
    '''
    - Aloud, M., Tsang, E., Olsen, R. & Dupuis, A. (2012). A Directional-Change Event Approach for Studying Financial Time Series. Economics, 6(1), 20120036.
    - data : a list or array-like time series object
    - d : theta value, which is a threshold to decide upturn/downturn event
    '''
    p = pd.DataFrame({
    "Price": data
    })
    p["Event"] = 0
    run = "upward" # initial run
    ph = p['Price'][0] # highest price
    pl = ph # lowest price
    pl_i = ph_i = 0

    for t in range(0, len(p)):
      pt = p["Price"][t]
      if run == "downward":
          if pt < pl:
              pl = pt
              pl_i = t
          if pt >= pl * (1 + d):
              p.at[pl_i, 'Event'] = +1
              run = "upward"
              ph = pt
              ph_i = t
                # print(">> {} - Upward! : {}%, value {}".format(pl_i, round((pt - pl)/pl, 2), round(pt - pl,2)))
      elif run == "upward":
        if pt > ph:
              ph = pt
              ph_i = t
        if pt <= ph * (1 - d):
              p.at[ph_i, 'Event'] = -1
              run = "downward"
              pl = pt
              pl_i = t
              # print(">> {} - Downward! : {}%, value {}".format(ph_i, round((ph - pt)/ph, 2), round(ph - pt,2)))

    return p







def get_non_zero_events(df):

  return df.index[df['Event'] != 0].tolist() 



def infer_process_data(file_name, symbols, dc_list, TOPERCENTAGE=100):

    # Read the entire CSV file
    close_prices_org = pd.read_csv(file_name, index_col=0, parse_dates=True)
    print(close_prices_org.shape)
    
    # Initialize a dictionary to store results for each chunk
    results_by_chunk = {}

    # Process the first chunk from the start (index 0) to the first index in dc_list
    if len(dc_list) > 0:
        start_row = 0
        end_row = dc_list[0]
        
        # Select the chunk of data
        close_prices = close_prices_org.iloc[start_row:end_row][symbols]
        names = close_prices.columns.to_list()

        # Clean the data by removing NaN values
        close_prices = close_prices.dropna(how='all')
        close_prices.fillna(method='ffill', inplace=True)
        close_prices.fillna(method='bfill', inplace=True)

        # Check if there are enough rows for calculation
        if close_prices.shape[0] < 2:
            print(f"Not enough data in chunk 0: {close_prices.shape}")
            return results_by_chunk

        close_prices_assets = close_prices

        # Calculate log returns
        returns = pd.DataFrame({})
        for name in names:
            current_returns = np.log(close_prices[name] / close_prices[name].shift(1))
            returns[name] = current_returns.iloc[1:] * TOPERCENTAGE  # Remove NaN from first row

        # Check if returns DataFrame is empty
        if returns.empty or len(returns) < 2:
            print(f"Not enough returns data in chunk 0: {returns.shape}")
            return results_by_chunk
        else:
            # Proceed with further calculations
            print(f"Returns DataFrame calculated successfully in chunk_{0}.")

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

        # Store results in the dictionary
        results_by_chunk['chunk_0'] = {
            'data_tensor': data_tensor,
            'returns_p': returns_p,
            'mean_return': mean_return,
            'cov': cov,
            'cov_np': cov_np,
            'data_w_ret': data_w_ret,
            'close_prices_assets': close_prices_assets,
            'names': names
        }

    # Process remaining chunks based on dc_list
    for i in range(1, len(dc_list)):
        start_row = dc_list[i - 1]
        end_row = dc_list[i]
        
        # Select the chunk of data
        close_prices = close_prices_org.iloc[start_row:end_row][symbols]
        names = close_prices.columns.to_list()

        # Clean the data by removing NaN values
        close_prices = close_prices.dropna(how='all')
        close_prices.fillna(method='ffill', inplace=True)
        close_prices.fillna(method='bfill', inplace=True)

        # Check if there are enough rows for calculation
        if close_prices.shape[0] < 2:
            print(f"Not enough data in chunk {i}: {close_prices.shape}")
            continue  # Skip this chunk

        close_prices_assets = close_prices

        # Calculate log returns
        returns = pd.DataFrame({})
        if close_prices.shape[0] < 2:
            print("Not enough data to calculate returns.")
        else:
            for name in names:
                # Calculate current returns
                current_returns = np.log(close_prices[name] / close_prices[name].shift(1))

                # Check if current_returns has enough valid entries
                if current_returns.isnull().all():
                    print(f"No valid returns for {name}.")
                    continue  # Skip this asset if no valid returns

                # Store the valid returns in the returns DataFrame
                returns[name] = current_returns.iloc[1:] * TOPERCENTAGE  # Remove NaN from first row

            # After populating returns, check if it's empty
            if returns.empty or returns.shape[0] < 2:
                print("Returns DataFrame is empty or has insufficient data.")
            else:
                # Proceed with further calculations
                print(f"Returns DataFrame calculated successfully in chunk_{i}.")

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

        # Store results in the dictionary
        results_by_chunk[f'chunk_{i}'] = {
            'data_tensor': data_tensor,
            'returns_p': returns_p,
            'mean_return': mean_return,
            'cov': cov,
            'cov_np': cov_np,
            'data_w_ret': data_w_ret,
            'close_prices_assets': close_prices_assets,
            'names': names
        }

    # Handle the last chunk if it doesn't have a next value in dc_list
    if len(dc_list) > 0 and dc_list[-1] < close_prices_org.shape[0]:
        start_row = dc_list[-1]
        end_row = close_prices_org.shape[0]  # Use the last row of the original DataFrame
        
        # Select the chunk of data
        close_prices = close_prices_org.iloc[start_row:end_row][symbols]
        names = close_prices.columns.to_list()

        # Clean the data by removing NaN values
        close_prices = close_prices.dropna(how='all')
        close_prices.fillna(method='ffill', inplace=True)
        close_prices.fillna(method='bfill', inplace=True)

        # Check if there are enough rows for calculation
        if close_prices.shape[0] < 2:
            print(f"Not enough data in last chunk: {close_prices.shape}")
            return results_by_chunk

        close_prices_assets = close_prices

        # Calculate log returns
        returns = pd.DataFrame({})
        for name in names:
            current_returns = np.log(close_prices[name] / close_prices[name].shift(1))
            returns[name] = current_returns.iloc[1:] * TOPERCENTAGE  # Remove NaN from first row

        # Check if returns DataFrame is empty
        if returns.empty or len(returns) < 2:
            print(f"Not enough returns data in last chunk: {returns.shape}")
            return results_by_chunk
        else:
            # Proceed with further calculations
            print(f"Returns DataFrame calculated successfully in chunk_{len(dc_list)}.")

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

        # Store results in the dictionary
        results_by_chunk[f'chunk_{len(dc_list)}'] = {
            'data_tensor': data_tensor,
            'returns_p': returns_p,
            'mean_return': mean_return,
            'cov': cov,
            'cov_np': cov_np,
            'data_w_ret': data_w_ret,
            'close_prices_assets': close_prices_assets,
            'names': names
        }

    return results_by_chunk






if __name__ == "__main__":

    Market_index = ['SPY']
    start_date = "2023-06-13"
    end_date = "2024-06-11"
    orig_merged_df = download_stock_data(Market_index, start=start_date, end=end_date)
    print(orig_merged_df)

    dc = DCfilter(orig_merged_df['SPY'], d=0.08)
    dc_list = get_non_zero_events(dc)
    print(dc_list)


    symbols = ['AAPL',   'AMGN','AMZN',    'BRK-B',   'C',   'COST', 'DHR', 'GE',  'GOOGL',  'IBM',  'JNJ',  'KO',  'LLY',  'MA',  'MCD',  'META',  'MSFT',  'NVDA',  'PEP',  'PG',  'RTX',  'SO',  'TSLA', 'UNH','USB','WFC', 'WMT', 'AMD','DIS','NFLX']

    results = infer_process_data('index.csv', symbols, dc_list)
    print(results.keys())





    # Initialize the model
    TOPERCENTAGE=100 #TOPERCENTAGE=1
    TREASURYRATEYEARLY=0.01
    BUSINESSDAYSINCALENDAR=252
    INVESTDURATION=252
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250, help="Number of epochs to train the model")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--dropout_rate', type=float, default=0.25, help="Dropout rate in the model")
    parser.add_argument('--num_heads', type=int, default=3, help="Number of attention heads")
    parser.add_argument('--ff_dim', type=int, default=64, help="Feed-forward dimension size")
    parser.add_argument('--num_encoder_layers', type=int, default=5, help="Number of encoder layers")
    parser.add_argument('--num_residual_layers', type=int, default=3, help="Number of residual layers")
    parser.add_argument('--block_size', type=int, default=32, help="Block size for splitting the data")  
    args = parser.parse_args()



    allocations_results = {}  # Dictionary to store allocations for each chunk
    print("\n")
    print(f"###################### Model load in {device} device for inference ######################")
    print("\n")

    for key in results.keys():


        # Extract the tensor for the current chunk
        test_data_tensor = results[key]['data_tensor']  
        returns_p = results[key]['returns_p']
        mean_return = results[key]['mean_return']
        cov = results[key]['cov']
        cov_np = results[key]['cov_np']
        data_w_ret = results[key]['data_w_ret']
        close_prices_assets = results[key]['close_prices_assets']
        names = results[key]['names']

        


        input_shape = (test_data_tensor.shape[1], test_data_tensor.shape[2])   
        output_size = len(close_prices_assets.columns)


        
        print(f"Using device: {device}")



        model = SSOPO(
            input_shape=input_shape,
            output_size=output_size,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout_rate,
            num_encoder_layers=args.num_encoder_layers,
            num_residual_layers=args.num_residual_layers
            
        ).to(device)  


        print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
        print(model)



        # Ensure the tensor is on the correct device
        test_data_tensor = test_data_tensor.to(device)





        # Train the model
        train_model(
        model,
        test_data_tensor,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
       )

        # Optionally print the allocations
        print("\n")
        print(f"######################### {key} ########################")
        

        allocations =   model(test_data_tensor.to(next(model.parameters()).device)).detach().cpu().numpy()[0]  
        print(f"Allocations for {key}: {allocations}")
        allocations_results[key] = allocations  
        best_w = allocations
        opt_risk = np.sqrt(best_w.dot(BUSINESSDAYSINCALENDAR * cov).dot(best_w))
        opt_ret = mean_return.dot(best_w) * BUSINESSDAYSINCALENDAR


        duration_days = 252
        single_asset_risks = np.sqrt(np.diag(cov_np))
        single_asset_returns = mean_return.values

        rsk = pd.Series(single_asset_risks, index=names, name='DAILY RISK %')
        rsk['OPT_PORTFOLIO'] = opt_risk
        rsk = rsk.round(2)


        rtr = pd.Series(single_asset_returns, index=names, name='CUMULATIVE RETURN %')
        rtr['OPT_PORTFOLIO'] = opt_ret
        rtr = (((1 + rtr / TOPERCENTAGE) ** duration_days - 1) * TOPERCENTAGE).round(2)


        wgh = pd.Series(best_w, index=names, name='WEIGHTS %')
        wgh['OPT_PORTFOLIO'] = 1
        wgh = (wgh * TOPERCENTAGE).round(2)

        result = pd.DataFrame(rtr).join(rsk).join(wgh)
        print(result)

        sharp_of_portfolio = opt_ret / opt_risk
        print("Portfolio Sharpe ratio:", sharp_of_portfolio)
        print(f"Portfolio Return: {opt_ret:.2f}%")
        print(f"Portfolio Volatility: {opt_risk:.2f}%")
        print(cov)
        
    print("\n")
    print(allocations_results)





















