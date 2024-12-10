import torch
import torch.nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("..")
from model.PO_MLP import *
import torch.optim as optim
from datahandler import *
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm





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
            progress_bar.update(1)  # Update the progress bar by 50 epochs
        
    writer.close()












def main():
    # Define symbols (example)
    symbols = ['AAPL',    'AMGN','AMZN',    'BRK-B',   'C',   'COST', 'DHR', 'GE',  'GOOGL',  'IBM',  'JNJ',  'KO',  'LLY',  'MA',  'MCD',  'META',  'MSFT',  'NVDA',  'PEP',  'PG',  'RTX',  'SO',  'TSLA', 'UNH','USB','WFC', 'WMT', 'AMD','DIS','NFLX']
    
    # Process and save train data
    data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names = process_data('index.csv', symbols)


 

    return data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names




if __name__ == "__main__":
    # Argument parsing for hyperparameters

    TOPERCENTAGE=100 #TOPERCENTAGE=1
    TREASURYRATEYEARLY=0.01
    BUSINESSDAYSINCALENDAR=252
    INVESTDURATION=252


    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train the model")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--dropout_rate', type=float, default=0.25, help="Dropout rate in the model")
    parser.add_argument('--block_size', type=int, default=8, help="Block size for splitting the data") 
    parser.add_argument('--temperature', type=int, default=0.5, help="temp")

    args = parser.parse_args()

    data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names = main()
    
    

 
    


    data_tensor = torch.tensor(data_w_ret, dtype=torch.float32)

    


    block_size = args.block_size
    sequence_length = data_tensor.shape[0]  

    


    trimmed_length = (sequence_length // block_size) * block_size



    data_tensor = data_tensor[:trimmed_length]


    num_blocks = trimmed_length // block_size



    data_tensor = data_tensor.view(num_blocks, block_size, data_tensor.shape[1])





    print("Train data shape:", data_tensor.shape)






    # Define input shape for the model based on the reshaped data
    input_shape = (block_size, data_tensor.shape[2])   
    output_size = len(close_prices_assets.columns)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    model = MLP(
        input_shape=input_shape,
        hidden_dim=64, num_layers=2, dropout=0.25
        
    ).to(device)  


    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    print(model)



    train_data = data_tensor.to(device)





    # Train the model
    train_model(
        model,
        train_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
)


    
    




    print("####################################################  Online Learning Mode:  #####################################################")

    




    allocations =   model(train_data.to(next(model.parameters()).device)).detach().cpu().numpy()[0]
    list_allocations = allocations.tolist()
    print(list_allocations)    
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
    







    
    


    
        

    
    
