import torch
import copy
import numpy as np
from generate_mdps import generate_mdp, value_iteration, find_policy


class GraphData(torch.utils.data.IterableDataset):
    def __init__(self,
                 num_states, num_actions, epsilon,
                 graph_type='random', seed=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = epsilon
        self.graph_type = graph_type
        self.seed = seed

    def build_graph(self):
        """
        p = torch.tensor([[[0., 1.], [0., 1.]]])
        r = torch.tensor([[0.1], [0.7]])
        discount = 0.1
        """
        p, r, discount = generate_mdp(num_states=self.num_states, num_actions=self.num_actions,
                                      graph_type=self.graph_type, seed=self.seed)

        vs = value_iteration(p=p, r=r, discount=discount, eps=self.eps)
        policy = find_policy(p, r, discount, vs[-1])
        policy_dict = {
            'p': p,
            'r': r,
            'discount': discount,
            'policy': policy,
        }

        p = torch.transpose(p, dim0=-1, dim1=-2)
        #print("Iterations ", vs.shape[0])
        # p: a, s, s'
        # r: s, a
        # discount: 1
        # vs: iter, s
        np.set_printoptions(threshold=np.infty)
        #print("VS ", vs.numpy())
        #exit(0)
        ones = torch.ones_like(p)
        zeros = torch.zeros_like(p)
        adj_mask = torch.where(p > 0, ones, zeros).unsqueeze(dim=-1)  # a, s, s', 1

        adj_mat_p = p.unsqueeze(dim=-1)  # a, s, s', 1
        discount_mat = torch.ones_like(adj_mat_p) * discount
        adj_mat = torch.cat((adj_mat_p, discount_mat), dim=-1)  # a, s, s, 2

        v_node_feat = vs.unsqueeze(dim=1).repeat(1, p.shape[0], 1)  # iter, a, s
        r_node_feat = r.transpose(dim0=0, dim1=1)  # a, s
        r_node_feat = r_node_feat.unsqueeze(dim=0).repeat(v_node_feat.shape[0], 1, 1)  # iter, a, s
        node_feat = torch.cat((v_node_feat.unsqueeze(dim=-1), r_node_feat.unsqueeze(dim=-1)), dim=-1)  # iter, a, s, 2

        # adj_mat_r = r.transpose(dim0=0, dim1=1) # a, s
        # adj_mat_r = adj_mat_r.unsqueeze(dim=-1).repeat(1, 1, self.num_states) # a, s, s
        # adj_mat_r = adj_mat_r.unsqueeze(dim=-1)
        # adj_mat = torch.cat((adj_mat_p, adj_mat_r), dim=-1)

        yield (node_feat, adj_mat, adj_mask, vs, policy_dict)

    def __iter__(self):
        return self.build_graph()
