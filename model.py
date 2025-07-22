import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=2, dropout_rate=0.3):
        super(CNN_GRU_Attention, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3) 
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout_gru = nn.Dropout(dropout_rate)
        
        # GRU
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 2 for bidirectional GRU
        self.softmax = nn.Softmax(dim=1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Using attention-weighted context
        
    def forward(self, x):
        # CNN Feature Extraction
        x = x.permute(0, 2, 1)  # Change to (batch, channels, time) for CNN
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout3(x)
        
        # Reshape for GRU
        x = x.permute(0, 2, 1)  # Change back to (batch, time, features) for GRU
        gru_out, _ = self.gru(x)  # Shape: (batch, time_steps, hidden_dim*2)
        gru_out = self.dropout_gru(gru_out)
        
        # Attention Mechanism
        attention_scores = self.attention(gru_out)  # Shape: (batch, time_steps, 1)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch, time_steps, 1)
        
        # Apply attention weights
        context_vector = torch.sum(gru_out * attention_weights, dim=1)  # Shape: (batch, hidden_dim*2)
        
        # Fully Connected Layer
        output = self.fc(context_vector)  # Output shape: (batch, output_dim)
        
        return output