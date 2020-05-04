import torch
import torch.nn as nn
import numpy as np

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations=1,  # number of relation types (adjacency matrices)
                 activation=None):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * n_relations, out_features=out_features)
        self.n_relations = n_relations

    def chebyshev_basis(self, L, X):
        # GCN
        return torch.bmm(L, X)  # B,N,1,F

    def laplacian_batch(self, A):
        print("A ", A.shape)
        batch, N = A.shape[:2]
        A_hat = A
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, N, 2) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, discount = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []

        for rel in range(self.n_relations):
            print("A ", A.shape)  # b, a, s, s, 2
            print(A[:, rel, :, :, :].shape)  # b, s, s, 2
            L = self.laplacian_batch(A[:, rel, :, :, :])
            print("L ", L.shape)
            x_hat.append(self.chebyshev_basis(L, x))
        x = self.fc(torch.cat(x_hat, 2))

        if self.activation is not None:
            x = self.activation(x)
        return (x, A, discount)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_relations=1):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                n_relations=n_relations,
                                                activation=nn.ReLU(inplace=True)) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)
        x = self.fc(x)
        return x


class MessagePassing(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 activation):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, out_features, bias=False))
        self.edge_proj = nn.Linear(2*out_features + edge_features, out_features)
        self.activation = activation

    def compute_adj_mat(self, A):
        batch, N = A.shape[:2]
        I = torch.eye(N).unsqueeze(0).to(cuda_device)
        return A + I

    def forward(self, data):
        x, adj = data
        x = self.node_proj(x)
        # a, s, out_features
        num_states = x.shape[1]
        x_i = x.unsqueeze(dim=2).repeat(1, 1, num_states, 1) # a, s, 1, out_features
        x_j = x.unsqueeze(dim=1).repeat(1, num_states, 1, 1) #a, 1, s', out_features
        messages = self.edge_proj(torch.cat((x_i, x_j, adj), dim=-1))
        # if self.activation is not None:
        #    messages = self.activation(messages)
        neighb = torch.sum(messages, dim=-2)
        new_x = neighb + x
        return (new_x, adj)


class MPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 out_features,
                 filters=[64, 64, 64]):
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
        x, adj = self.mps(data)

        x, ind = torch.max(x, dim=0)
        x = self.fc(x)
        return x