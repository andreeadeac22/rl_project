import torch


def generate_mdp(num_states, num_actions):
    # P: a, s, s'
    # R: s, a

    p = torch.rand(num_actions, num_states, num_states)
    r = torch.rand(num_states, num_actions) * (-1. - 1.) + 1. # between -1, 1

    mask = torch.randint(0, 2, (num_actions, num_states, num_states))

    p = p * mask

    print("p ", p)
    as_sum = torch.sum(p, dim=-1, keepdim=True)
    print("as_sum ", as_sum)
    print("as_sum ", as_sum.shape)

    p = p / as_sum

    print("p ", p)
    print("r ", r)
    return p, r


generate_mdp(6, 2)