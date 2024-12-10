import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("..")




class SSOPO(nn.Module):
    def __init__(self, input_shape, output_size, num_heads=2, ff_dim=32, dropout=0.25, num_encoder_layers=4, num_residual_layers=None):
        super(SSOPO, self).__init__()
        print(f"d_model: {input_shape[1]}, num_heads: {num_heads}, num_encoder_layers: {num_encoder_layers}")

        # Set default for num_residual_layers if not specified
        if num_residual_layers is None:
            num_residual_layers = num_encoder_layers

        assert num_residual_layers <= num_encoder_layers, \
            "num_residual_layers cannot exceed num_encoder_layers"

        self.num_residual_layers = num_residual_layers

        # Define Transformer Encoder layers
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_shape[1], nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Additional layers for non-linearity and regularization
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Pooling and output layers
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(input_shape[1], output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = x  # Initial residual connection

        for idx, layer in enumerate(self.encoder_blocks):
            if idx < self.num_residual_layers:
                x = layer(x) + residual  # Apply residual connection
                residual = x  # Update residual for the next layer
            else:
                x = layer(x)  # No residual connection for remaining layers
            
            # Apply gelu activation and Dropout after each layer
            x = self.gelu(x)
            
        
        # Pooling and output layer
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        allocations = self.softmax(self.output_layer(x))
        return allocations

    def sharpe_loss(self, y_pred, data):
        # Price normalization
        prices = data[:, :, :y_pred.size(1)]  
        prices_normalized = prices / prices[:, 0:1, :]
        
        # Portfolio calculations
        y_pred = y_pred.unsqueeze(1)
        portfolio_values = torch.sum(prices_normalized * y_pred, dim=2)
        portfolio_returns = (portfolio_values[:, 1:] - portfolio_values[:, :-1]) / portfolio_values[:, :-1]
        sharpe_ratio = torch.mean(portfolio_returns) / torch.std(portfolio_returns)
        sharpe_ratio = sharpe_ratio
        
        return -sharpe_ratio





