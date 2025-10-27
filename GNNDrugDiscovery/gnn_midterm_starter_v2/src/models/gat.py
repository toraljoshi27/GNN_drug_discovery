
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, num_layers, dropout, out_dim, task):
        super().__init__()
        self.task = task
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        cur_dim = hidden_dim * heads

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(cur_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
            cur_dim = hidden_dim * heads

        self.head = nn.Linear(cur_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # ---- CRITICAL: ensure float features for the linear layers inside GAT ----
        if not torch.is_floating_point(x):
            x = x.float()
        # -------------------------------------------------------------------------

        for conv in self.convs:
            x = conv(x, edge_index)     # no edge_attr used by default in GATv2Conv
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.head(x)
        return out