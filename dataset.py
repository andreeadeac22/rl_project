import torch
import copy
import numpy as np
from generate_mdps import generate_mdp, value_iteration


class GraphData(torch.utils.data.IterableDataset):
    def __init__(self,
                 num_states, num_actions, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = epsilon

    def build_graph(self):
        p, r, discount = generate_mdp(num_states=self.num_states, num_actions=self.num_actions)
        vs = value_iteration(p=p, r=r, discount=discount, eps=self.eps)
        yield (p, r, discount, vs)


    def __iter__(self):
        return self.build_graph()