import torch
import torch.nn as nn
import numpy as np

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MessagePassing(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 activation):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, out_features, bias=False))
        self.edge_proj = nn.Linear(edge_features, out_features)
        self.message_proj = nn.Linear(3*out_features, out_features)
        self.activation = activation

    def compute_adj_mat(self, A):
        batch, N = A.shape[:2]
        I = torch.eye(N).unsqueeze(0).to(cuda_device)
        return A + I

    def forward(self, data):
        x, adj, adj_mask = data
        x = self.node_proj(x)
        adj = self.edge_proj(adj)
        # a, s, out_features
        num_states = x.shape[1]
        x_i = x.unsqueeze(dim=2).repeat(1, 1, num_states, 1)  # a, s, 1, out_features
        x_j = x.unsqueeze(dim=1).repeat(1, num_states, 1, 1)  # a, 1, s', out_features
        messages = self.message_proj(torch.cat((x_i, x_j, adj), dim=-1))
        messages = messages * adj_mask
        if self.activation is not None:
            messages = self.activation(messages)
        neighb = torch.sum(messages, dim=-2)
        #neighb, ind = torch.max(messages, dim=-2)
        #neighb = torch.mean(messages, dim=-2)
        new_x = neighb + x
        return (new_x, adj, adj_mask)


class MPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 filters=[64]):
        super().__init__()
        self.mps = nn.Sequential(*([
            MessagePassing(node_features=node_features if layer == 0 else filters[layer - 1],
                           edge_features=edge_features,
                            out_features=f,
                            activation=nn.ReLU(inplace=True)) for layer, f in enumerate(filters)]))
        self.fc = nn.Linear(in_features=filters[-1], out_features=out_features)

    def forward(self, data):
        # node.shape: a, s, 2
        # adj.shape: a, s, s, 2
        x, adj, adj_mask = self.mps(data)

        x, ind = torch.max(x, dim=0)
        #x = torch.sum(x, dim=0)
        x = self.fc(x)
        return x