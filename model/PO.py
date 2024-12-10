import torch
import torch.nn as nn
import math
import sys
import numpy as np
sys.path.append("..")



class SSOPO(nn.Module):
    def __init__(self, input_shape, output_size, num_heads=2, ff_dim=32, dropout=0.25, num_encoder_layers=4):
        super(SSOPO, self).__init__()
        print(f"d_model: {input_shape[1]}, num_heads: {num_heads}, num_encoder_layers: {num_encoder_layers}")


        self.encoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=input_shape[1], nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_encoder_layers)])
        

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(input_shape[1], output_size)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):

        
        

        x = self.encoder_blocks(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        allocations = self.softmax(self.output_layer(x))
        return allocations


    def sharpe_loss(self, y_pred, data):
 
        
        prices = data[:, :, :y_pred.size(1)]  
        prices_normalized = prices / prices[:, 0:1, :]
        y_pred = y_pred.unsqueeze(1)
        portfolio_values = torch.sum(prices_normalized * y_pred, dim=2)
        portfolio_returns = (portfolio_values[:, 1:] - portfolio_values[:, :-1]) / portfolio_values[:, :-1]
        sharpe_ratio = torch.mean(portfolio_returns) / torch.std(portfolio_returns)



        return -sharpe_ratio









