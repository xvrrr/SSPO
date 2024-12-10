import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("..")





class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, num_layers=4, dropout=0.25):
        super(MLP, self).__init__()

        print(f"Input Shape: {input_shape}, Hidden Dim: {hidden_dim}, Number of Layers: {num_layers}")

        # Calculate the flattened input size
        flattened_input_size = input_shape[0] * input_shape[1]  # sequence_length * num_assets

        # Define MLP layers
        self.mlp1 = nn.Linear(flattened_input_size, hidden_dim)  # First hidden layer
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.mlp3 = nn.Linear(hidden_dim, 30)  # Output layer (for allocation probabilities)

        # Softmax for classification outputs
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)  # batch_size
        
        # Flatten the input data: [batch_size, sequence_length * num_assets]
        flattened_input_size = x.size(1) * x.size(2)  # sequence_length * num_assets
        x = x.reshape(batch_size, flattened_input_size)  # Use reshape instead of view
        
        # Pass through MLP layers with ReLU activation
        x = torch.relu(self.mlp1(x))  # First hidden layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.mlp2(x))  # Second hidden layer with ReLU activation
        x = self.mlp3(x)  # Output layer
        
        # Softmax to get the allocation probabilities
        allocations = self.softmax(x)
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
        
        return -sharpe_ratio
