
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, heads=4, num_layers=3, dropout=0.2, out_dim=1, task="binary"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim//heads, heads=heads, dropout=dropout))
        for _ in range(num_layers-1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads, dropout=dropout))
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, out_dim)
        self.task = task

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)
