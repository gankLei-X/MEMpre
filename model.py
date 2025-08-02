import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, dim, n_heads, dim_feedforward, num_layers, seq_length,
                 input_size, hidden_sizes, num_classes):
        super(TransformerModel, self).__init__()
        # x1分支
        self.pos_encoder = PositionalEncoding(dim, max_len=seq_length)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.transformer2 = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(40, 8)
        # x2分支
        MLPlayers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            MLPlayers.append(nn.Linear(in_size, hidden_size))
            MLPlayers.append(nn.ReLU())
            in_size = hidden_size
        MLPlayers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*MLPlayers)
        self.output = nn.Linear(16, 8)
        self.softmax = nn.Softmax(dim = 1)

    def forward_transformer(self, x1):
        x1 = x1.permute([0, 2, 1])
        x1 = self.pos_encoder(x1)
        x1 = self.transformer(x1)
        x1_1 = self.transformer2(x1)
        x1 = x1 + x1_1
        x1 = torch.mean(x1, dim=1)
        x1 = self.fc(x1)
        return x1

    def forward_mlp(self, x2):
        x2 = self.network(x2)
        return x2

    def forward(self, x1, x2):
        x1 = self.forward_transformer(x1)
        x2 = self.forward_mlp(x2)
        output = torch.cat([x1, x2], dim=1)
        output = self.output(output)
        return output