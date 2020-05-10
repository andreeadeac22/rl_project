import networkx as nx
import numpy as np
import torch
import math


def erdos_renyi(N, degree, seed):
    """ Creates an Erdős-Rényi or binomial graph of size N with degree/N probability of edge creation """
    return nx.fast_gnp_random_graph(N, degree / N, seed, directed=False)


def barabasi_albert(N, degree, seed):
    """ Creates a random graph according to the Barabási–Albert preferential attachment model
        of size N and where nodes are attached with degree edges """
    return nx.barabasi_albert_graph(N, degree, seed)


def grid(N):
    """ Creates a m x k 2d grid graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.grid_2d_graph(m, N // m)


def caveman(N):
    """ Creates a caveman graph of m cliques of size k, with m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.caveman_graph(m, N // m)


def tree(N, seed):
    """ Creates a tree of size N with a power law degree distribution """
    return nx.random_powerlaw_tree(N, seed=seed, tries=10000)


def ladder(N):
    """ Creates a ladder graph of N nodes: two rows of N/2 nodes, with each pair connected by a single edge.
        In case N is odd another node is attached to the first one. """
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def line(N):
    """ Creates a graph composed of N nodes in a line """
    return nx.path_graph(N)


def star(N):
    """ Creates a graph composed by one center node connected N-1 outer nodes """
    return nx.star_graph(N - 1)


def caterpillar(N, seed):
    """ Creates a random caterpillar graph with a backbone of size b (drawn from U[1, N)), and N − b
        pendent vertices uniformly connected to the backbone. """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, seed):
    """ Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
        from U[1, N − b ]) pendent vertices uniformly connected to the backbone, and additional
        N − b − p pendent vertices uniformly connected to the previous pendent vertices """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))
    return G


def process(file='gridworld_8x8.npz', train=False):
    with np.load(file, mmap_mode='r') as f:
        if train:
            images = f['arr_0']
        else:
            images = f['arr_4']
    images = images.astype(np.float32)

    nb_images = images.shape[0]
    nb_actions = 8
    nb_states = 64

    dx = [1, 0, -1, 0, 1, 1, -1, -1]
    dy = [0, 1, 0, -1, 1, -1, 1, -1]
    r = [-0.1, -0.1, -0.1, -0.1, -0.1414, -0.1414, -0.1414, -0.1414]

    #Ps = []
    #Rs = []

    # Print number of samples
    """
    if train:
        print("Number of Train Samples: {0}".format(images.shape[0]))
    else:
        print("Number of Test Samples: {0}".format(images.shape[0]))
    """
    img_index = np.random.randint(images.shape[0])

    #for img in indices:
    grid = images[img_index, 0]
    reward = images[img_index, 1] / 10.0

    ind = []
    rev_map = {}

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                rev_map[(x, y)] = len(ind)
                ind.append((x, y))

    nb_states = len(ind)

    P = torch.zeros((nb_actions, nb_states, nb_states))
    R = torch.zeros((nb_states, nb_actions))

    for s, (x, y) in enumerate(ind):
        for act in range(nb_actions):
            if reward[x, y] > 0.0:
                P[act][s][s] = 1.0
            else:
                next_x = x + dx[act]
                next_y = y + dy[act]
                if (next_x, next_y) not in rev_map:
                    next_x = x
                    next_y = y
                next_r = r[act] + reward[next_x, next_y]
                s_prime = rev_map[(next_x, next_y)]
                P[act][s][s_prime] = 1.0
                # train on similar R distributions?!
                # R[s][act] = next_r * 2.5

    R = torch.rand(nb_states, nb_actions) * (-1. - 1.) + 1.

    #Ps.append(P)
    #Rs.append(R)
    return P, R

"""
Ps, Rs = process(file='gridworld_28x28.npz', train=False)
lengths = [p.shape[1] for p in Ps]
print(min(lengths), max(lengths))
"""
