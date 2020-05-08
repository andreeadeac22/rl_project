import torch
import torch.nn as nn
import numpy as np

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MessagePassing(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 message_function='mpnn',
                 neighbour_state_aggr='sum',
                 state_residual_update='sum',
                 activation=None):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, out_features, bias=False))
        self.edge_proj1 = nn.Linear(edge_features, out_features)
        # self.edge_proj2 = nn.Linear(out_features, out_features)
        self.leakyrelu = nn.LeakyReLU()

        self.message_function = message_function
        if message_function == 'mpnn':
            self.message_proj = nn.Linear(3 * out_features, out_features)
        elif message_function == 'attention':
            self.attn_coeff = nn.Linear(2*out_features, 1)
            self.softmax = nn.Softmax()
        self.activation = activation
        self.neighbour_state_aggr = neighbour_state_aggr
        self.state_residual_update = state_residual_update
        self.predict_fc = nn.Linear(2*out_features, out_features)

    def compute_adj_mat(self, A):
        batch, N = A.shape[:2]
        I = torch.eye(N).unsqueeze(0).to(cuda_device)
        return A + I

    def forward(self, data):
        x, adj, adj_mask = data
        x = self.node_proj(x)
        adj = self.edge_proj1(adj)
        # adj = self.relu(adj)
        # adj = self.edge_proj2(adj)
        # a, s, out_features
        num_states = x.shape[1]
        x_i = x.unsqueeze(dim=2).repeat(1, 1, num_states, 1)  # a, s, 1, out_features
        x_j = x.unsqueeze(dim=1).repeat(1, num_states, 1, 1)  # a, 1, s', out_features

        if self.message_function == 'mpnn':
            messages = self.message_proj(torch.cat((x_i, x_j, adj), dim=-1))
            messages = messages * adj_mask

            if self.activation is not None:
                messages = self.activation(messages)

            if self.neighbour_state_aggr == 'sum':
                neighb = torch.sum(messages, dim=-2)
            elif self.neighbour_state_aggr == 'max':
                neighb, ind = torch.max(messages, dim=-2)
            elif self.neighbour_state_aggr == 'mean':
                neighb = torch.mean(messages, dim=-2)

        elif self.message_function == 'attention':
            a = self.attn_coeff(torch.cat((x_i, x_j), dim=-1))
            a = self.leakyrelu(a)
            a = (adj_mask - 1.)*1e9 + a
            alpha = self.softmax(a).squeeze(dim=-1)
            print("alpha ", alpha)
            neighb = torch.bmm(alpha, x)

        if self.state_residual_update == 'sum':
            new_x = neighb + x
        elif self.state_residual_update == 'concat':
            new_x = self.predict_fc(torch.cat((neighb, x), dim=-1))
        elif self.state_residual_update == 'neighb':
            new_x = neighb

        return (new_x, adj, adj_mask)


class MPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 message_function='mpnn',
                 neighbour_state_aggr='sum',
                 state_residual_update='sum',
                 action_aggr='max',
                 filters=[64]):
        super().__init__()
        self.mps = nn.Sequential(*([
            MessagePassing(node_features=node_features if layer == 0 else filters[layer - 1],
                           edge_features=edge_features if layer == 0 else filters[layer-1],
                           out_features=f,
                           message_function=message_function,
                           neighbour_state_aggr=neighbour_state_aggr,
                           state_residual_update=state_residual_update,
                           activation=nn.ReLU(inplace=True)) for layer, f in enumerate(filters)]))
        self.fc = nn.Linear(in_features=filters[-1], out_features=out_features)
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
        x = self.fc(x)
        return x
