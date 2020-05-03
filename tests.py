from generate_mdps import *
import numpy as np
import torch


def test_ift6760example():
    num_states = 2
    num_actions = 2

    p = torch.Tensor(np.array([[[0.75, 0.25], [0.2, 0.8]], [[0.99, 0.01], [0.8, 0.2]]]))
    r = torch.ones(num_states, num_actions)

    discount = 0.9
    v_0 = torch.zeros(num_states)
    vs = value_iteration(p, r, discount, v_0)

    assert np.allclose(vs[-1], [1 / (1 - discount)] * num_states), "ift6760 value iteration example test fails"


def test_generate_mdp():
    num_states = 2
    num_actions = 2
    p, r, discount = generate_mdp(num_states, num_actions)
    v_0 = torch.zeros(num_states)
    vs = value_iteration(p, r, discount, v_0)
    assert not torch.isnan(vs).any()


test_ift6760example()
test_generate_mdp()