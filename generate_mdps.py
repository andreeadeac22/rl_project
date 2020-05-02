import torch


def generate_mdp(num_states, num_actions, discount=0.9):
    # P: a, s, s'
    # R: s, a
    #
    attempt_no = 0
    while True:
        p = torch.rand(num_actions, num_states, num_states)
        r = torch.rand(num_states, num_actions) * (-1. - 1.) + 1.  # between -1, 1

        mask = torch.randint(0, 2, (num_actions, num_states, num_states))
        p = p * mask
        as_sum = torch.sum(p, dim=-1, keepdim=True)
        p = p / as_sum

        if not torch.isnan(p).any():
            return p, r, discount
        attempt_no += 1
        print("Attempts required to generate non-NaN transition matrix ", attempt_no)


def bellman_optimality_operator(v, P, R, discount):
    pv = torch.einsum('ijk,k->ji', P, v)
    newv, _ = torch.max(torch.add(R, discount * pv), dim=1)
    return newv


def value_iteration(v0, p, r, discount, eps=1e-8):
    iter_diff = float("inf")
    v_prev = v0
    while iter_diff > eps:
        newv = bellman_optimality_operator(v_prev, p, r, discount)
        iter_diff = torch.norm(newv - v_prev)
        # print("iter_diff ", iter_diff)
        v_prev = newv
    return newv

