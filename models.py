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

    def chebyshev_basis(self, L, X, K):
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.cat(Xt, dim=2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X)  # B,N,1,F

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        A_hat = A
        if self.K < 2 or self.scale_identity:
            I = torch.eye(N).unsqueeze(0).to(cuda_device)
            if self.scale_identity:
                I = 2 * I  # increase weight of self connections
            if self.K < 2:
                A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []

        for rel in range(self.n_relations):
            L = self.laplacian_batch(A[:, :, :, rel])
            x_hat.append(self.chebyshev_basis(L, x, self.K))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers

        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


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
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x
