import torch
import torch.nn as nn
import numpy as np


class MessagePassing(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_dim,
                 message_function=None,
                 message_function_depth=None,
                 neighbour_state_aggr=None):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=False))
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        self.message_function = message_function
        self.message_function_depth = message_function_depth

        if message_function == 'mpnn':
            self.message_proj1 = nn.Linear(3 * hidden_dim, hidden_dim)
            self.relu = nn.ReLU()
            if message_function_depth == 2:
                self.message_proj2 = nn.Linear(hidden_dim, hidden_dim)

        elif message_function == 'attention':
            self.attn_coeff = nn.Linear(2 * hidden_dim, 1)
            self.leakyrelu = nn.LeakyReLU()
            self.softmax = nn.Softmax(dim=1)
        self.neighbour_state_aggr = neighbour_state_aggr

    def forward(self, data):
        x, adj, adj_mask = data
        x = self.node_proj(x)
        adj = self.edge_proj(adj)

        num_states = x.shape[1]
        x_i = x.unsqueeze(dim=2).repeat(1, 1, num_states, 1)  # a, s, 1, out_features
        x_j = x.unsqueeze(dim=1).repeat(1, num_states, 1, 1)  # a, 1, s', out_features

        if self.message_function == 'mpnn':
            messages = self.message_proj1(torch.cat((x_i, x_j, adj), dim=-1))
            if self.message_function_depth == 2:
                messages = self.relu(messages)
                messages = self.message_proj2(messages)

            messages = messages * adj_mask
            messages = self.relu(messages)

            if self.neighbour_state_aggr == 'sum':
                neighb = torch.sum(messages, dim=-2)
            elif self.neighbour_state_aggr == 'max':
                neighb, ind = torch.max(messages, dim=-2)
            elif self.neighbour_state_aggr == 'mean':
                neighb = torch.mean(messages, dim=-2)
            else:
                raise NotImplementedError

        elif self.message_function == 'attention':
            a = self.attn_coeff(torch.cat((x_i, x_j), dim=-1))
            a = self.leakyrelu(a)
            a = (adj_mask - 1.) * 1e9 + a
            alpha = self.softmax(a).squeeze(dim=-1)
            neighb = torch.bmm(alpha, x)
        else:
            raise NotImplementedError

        new_x = neighb + x
        return (new_x, adj, adj_mask)


class MPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_dim=None,
                 out_features=None,
                 message_function=None,
                 message_function_depth=None,
                 neighbour_state_aggr=None):
        super().__init__()
        self.mps = MessagePassing(node_features=node_features,
                                  edge_features=edge_features,
                                  hidden_dim=hidden_dim,
                                  message_function=message_function,
                                  message_function_depth=message_function_depth,
                                  neighbour_state_aggr=neighbour_state_aggr)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_features)

    def forward(self, data):
        # node.shape: a, s, 2
        # adj.shape: a, s, s, 2
        x, adj, adj_mask = self.mps(data)
        x, ind = torch.max(x, dim=0)
        x = self.fc(x)
        return x
