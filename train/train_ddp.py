import torch
import torch.nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import sys
sys.path.append("..")
from model.PO_DL import *
import torch.optim as optim
from datahandler import *
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist






def train_model(model, train_data, valid_data, epochs=850, learning_rate=0.001, batch_size=64, momentum = 0.8, log_dir='runs/experiment0', local_rank=0):

    writer = SummaryWriter(log_dir=log_dir)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  

     # Create DistributedSampler to split the data across processes (GPUs)
    train_sampler = DistributedSampler(train_data, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)
 

    # Creating PyTorch datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    

    progress_bar = tqdm(total=epochs, desc='Training Progress', unit='epoch', position=0, ncols=200)

    for epoch in range(epochs):
        # Training phase
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

    

        for batch in train_dataloader:
            batch_data = batch[0]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            allocations = model(batch_data)

            
            
            # Compute loss
            loss = model.module.sharpe_loss(allocations, batch_data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            

        

        # Normalize losses by number of batches
        train_loss /= len(train_dataloader)
        



        writer.add_scalar('Loss/train', train_loss, epoch)
     


        if epoch % 1 == 0:
            progress_bar.set_postfix({
                'Train Loss': train_loss,
            })
            progress_bar.update(1)  # Update the progress bar by 50 epochs



        
    progress_bar.close()
    writer.close()








def save_model(state_dict, path):
    torch.save(state_dict, path)
    print(f'Model saved to {path}')



def get_data(args):
    # Define symbols (example)
    
    symbols = ['AAPL',  'AMGN','AMZN',    'BRK-B',   'C',   'COST', 'DHR', 'GE',  'GOOGL',  'IBM',  'JNJ',  'KO',  'LLY',  'MA',  'MCD',  'META',  'MSFT',  'NVDA',  'PEP',  'PG',  'RTX',  'SO',  'TSLA', 'UNH','USB','WFC', 'WMT', 'AMD','DIS','NFLX']
    # Process and save train data

    



    data_tensor, returns_p, mean_return, cov, cov_np, data_w_ret, close_prices_assets, names = process_data('index.csv', symbols)
    torch.save(data_tensor, 'concat_asset.pt')


    


    data_tensor = torch.tensor(data_w_ret, dtype=torch.float32)



    


    block_size = args.block_size
    sequence_length = data_tensor.shape[0]  
    

    


    trimmed_length = (sequence_length // block_size) * block_size
   



    data_tensor = data_tensor[:trimmed_length]




    num_blocks = trimmed_length // block_size



    data_tensor = data_tensor.view(num_blocks, block_size, data_tensor.shape[1])


    print("Train data shape:", data_tensor.shape)



    input_shape = (block_size, data_tensor.shape[2])   
    output_size = len(close_prices_assets.columns)

    return data_tensor, input_shape, output_size

def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)                 # For PyTorch CPU
    torch.cuda.manual_seed(seed)           # For PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # For multi-GPU
    np.random.seed(seed)                   # For NumPy
    random.seed(seed)                      # For Python random module
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility
    torch.backends.cudnn.benchmark = False    # Disables benchmark for reproducibility


def main(args):
    # Set seed for reproducibility
    seed = args.seed if hasattr(args, 'seed') else 42  # Default seed is 42
    set_seed(seed)

    world_size = args.gpu
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=local_rank)  

    data_tensor, input_shape, output_size = get_data(args)

    model = SSOPO(
        input_shape=input_shape,
        output_size=output_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout_rate,
        num_encoder_layers=args.num_encoder_layers,
        num_residual_layers=args.rc_layers
    ).to(local_rank)



    print(model)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # Wrap the model in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_data = data_tensor.to(local_rank)
  

    # Train the model
    train_model(
        model,
        train_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        local_rank=local_rank,
        momentum = args.momentum
    )

    # Save model (only on rank 0)
    if local_rank == 0:
        save_model(model.module.state_dict(), 'model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000, help="Number of epochs to train the model")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--dropout_rate', type=float, default=0.25, help="Dropout rate in the model")
    parser.add_argument('--num_heads', type=int, default=2, help="Number of attention heads")
    parser.add_argument('--ff_dim', type=int, default=64, help="Feed-forward dimension size")
    parser.add_argument('--num_encoder_layers', type=int, default=5, help="Number of encoder layers")
    parser.add_argument('--block_size', type=int, default=64, help="Block size for splitting the data") 
    parser.add_argument('--temperature', type=int, default=0.5, help="Temperature")
    parser.add_argument('--momentum', type=int, default=0.95, help="Momentum")
    parser.add_argument('--gpu', type=int, default=2, help="GPU_num")
    parser.add_argument('--rc_layers', type=int, default=5, help="residual connection layers")
    parser.add_argument('--log_dir', type=str, default='runs/experiment0', help="Directory for TensorBoard logs")
    parser.add_argument('--seed', type=int, default=123, help="seed")

    
    


    args = parser.parse_args()
    main(args)
    
    
 








    
    


    
        

    
    
