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
                 neighbour_state_aggr=None,
                 state_residual_update='sum',
                 activation=None):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=False))
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        self.message_function = message_function
        self.message_function_depth = message_function_depth

        if message_function == 'mpnn':
            self.message_proj1 = nn.Linear(3 * hidden_dim, hidden_dim)
            if message_function_depth == 2:
                self.relu = nn.ReLU()
                self.message_proj2 = nn.Linear(hidden_dim, hidden_dim)

        elif message_function == 'attention':
            self.attn_coeff = nn.Linear(2 * hidden_dim, 1)
            self.leakyrelu = nn.LeakyReLU()
            self.softmax = nn.Softmax(dim=1)

        self.activation = activation
        self.neighbour_state_aggr = neighbour_state_aggr
        self.state_residual_update = state_residual_update

        if self.state_residual_update == 'concat':
            self.predict_fc = nn.Linear(2 * hidden_dim, hidden_dim)

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

            if self.activation is not None:
                messages = self.activation(messages)

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

        if self.state_residual_update == 'sum':
            new_x = neighb + x
        elif self.state_residual_update == 'concat':
            new_x = self.predict_fc(torch.cat((neighb, x), dim=-1))
        elif self.state_residual_update == 'neighb':
            new_x = neighb
        else:
            raise NotImplementedError

        return (new_x, adj, adj_mask)


class MPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_dim=None,
                 out_features=None,
                 message_function=None,
                 message_function_depth=None,
                 neighbour_state_aggr=None,
                 state_residual_update='sum',
                 action_aggr='max'):
        super().__init__()
        self.mps = MessagePassing(node_features=node_features,
                                  edge_features=edge_features,
                                  hidden_dim=hidden_dim,
                                  message_function=message_function,
                                  message_function_depth=message_function_depth,
                                  neighbour_state_aggr=neighbour_state_aggr,
                                  state_residual_update=state_residual_update,
                                  activation=nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_features)
        self.action_aggr = action_aggr

    def forward(self, data):
        # node.shape: a, s, 2
        # adj.shape: a, s, s, 2
        x, adj, adj_mask = self.mps(data)
        if self.action_aggr == 'max':
            x, ind = torch.max(x, dim=0)
        elif self.action_aggr == 'sum':
            x = torch.sum(x, dim=0)
        elif self.action_aggr == 'mean':
            x = torch.mean(x, dim=0)
        else:
            raise NotImplementedError
        x = self.fc(x)
        return x
