import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter

from models import *
from dataset import *
from generate_mdps import find_policy


def loss_fn(output, target, reduction='mean'):
    loss = (target.float().squeeze() - output.squeeze()) ** 2
    return loss.sum() if reduction == 'sum' else loss.mean()


def train(data):
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0
    for i in range(len(data) - 1):
        data[i] = data[i].to(args.device)
    node_feat, adj_mat, adj_mask, vs, policy_dict = data
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]
    last_loss = 0
    for step in range(iteration_steps - 1):
        optimizer.zero_grad()
        output = model((node_feat[step], adj_mat, adj_mask))
        loss = loss_fn(output, vs[step + 1] - vs[step])

        loss.backward()
        optimizer.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)

    print('Train Epoch: {}, samples: {} \t Last step loss {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
        epoch + 1, n_samples, loss.item(), train_loss / n_samples, time_iter))


def test(data):
    model.eval()
    start = time.time()
    test_loss, correct, n_samples = 0, 0, 0
    for i in range(len(data) - 1):
        data[i] = data[i].to(args.device)

    node_feat, adj_mat, adj_mask, vs, policy_dict = data
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]
    input_node_feat = node_feat[0]  # a,s,2

    values = torch.zeros(node_feat.shape[2], 1)
    accs = []
    gt_accs = []
    losses = []
    gt_losses = []
    for step in range(iteration_steps - 1):
        output = model((input_node_feat, adj_mat, adj_mask))
        values += output

        # output: s, 1 -> a,s,1
        # print("vs delta, Loss ", vs[step+1]-vs[step], loss.item())
        input_node_feat = torch.cat((output.unsqueeze(dim=0).repeat(args.test_num_actions, 1, 1) +  # a, s, 1
                                     input_node_feat[:, :, 0:1],
                                     node_feat[step + 1, :, :, 1:2]), dim=-1)

        losses += [loss_fn(values, vs[-1]).item()]
        gt_losses += [loss_fn(vs[step], vs[-1]).item()]

        gt_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'], vs[step])
        gt_accs += [100. * torch.eq(gt_policy, policy_dict['policy']).sum() / len(output)]
        predicted_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'], values.squeeze())
        accs += [100. * torch.eq(predicted_policy, policy_dict['policy']).sum() / len(output)]

    print('Test set (epoch {}): \t Last step accuracy {} \t Last step loss {:.6f} , Average loss: {:.6f} \n'.format(
        epoch + 1,
        accs[-1],
        losses[-1],
        np.mean(np.array(losses))))
    return losses[-1], accs[-1], losses, accs, gt_losses, gt_accs


parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('--train_num_states', type=int, default=20, help='number of states')
parser.add_argument('--train_num_actions', type=int, default=5, help='number of actions')

parser.add_argument('--test_num_actions', type=int, default=5, help='number of actions')

parser.add_argument('--epsilon', type=float, default=1e-8, help='termination condition (difference between two '
                                                                'consecutive values)')

parser.add_argument('--filters', type=int, default=[64], help='Hidden dim for node')

parser.add_argument('--patience', type=int, default=20)

parser.add_argument('--num_train_graphs', type=int, default=100, help='Number of graphs used for training')
parser.add_argument('--num_test_graphs', type=int, default=40, help='Number of graphs used for testing')

parser.add_argument('--lr', type=float, default=0.005, help='learning rate')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MPNN(node_features=2,
             edge_features=2,
             out_features=1,
             filters=args.filters).to(args.device)

print('\nInitialize model')
print(model)
train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
optimizer = optim.Adam(train_params, lr=args.lr)

iterable_train_dataset = GraphData(num_states=args.train_num_states, num_actions=args.train_num_actions,
                                   epsilon=args.epsilon)
train_loader = torch.utils.data.DataLoader(iterable_train_dataset, batch_size=None)
""""
for epoch in range(args.num_train_graphs):
    train(next(iter(train_loader)))

torch.save(model.state_dict(), 'mpnn.pt')
"""
model.load_state_dict(torch.load('mpnn.pt'))

import pickle

num_states = [20, 50, 100]
num_actions = [5, 10, 20]

for states in num_states:
    for actions in num_actions:
        args.test_num_actions = actions
        iterable_test_dataset = GraphData(num_states=states, num_actions=actions, epsilon=args.epsilon)
        test_loader = torch.utils.data.DataLoader(iterable_test_dataset, batch_size=None)

        test_last_losses = []
        test_all_losses = []
        test_last_accs = []
        test_all_accs = []
        all_gt_losses = []
        all_gt_accs = []

        for epoch in range(args.num_test_graphs):
            last_loss, last_acc, losses, accs, gt_losses, gt_accs = test(next(iter(test_loader)))
            test_last_losses += [last_loss]
            test_last_accs += [last_acc]
            test_all_losses += [losses]
            test_all_accs += [accs]
            all_gt_losses += [gt_losses]
            all_gt_accs += [gt_accs]
        print("States {}, actions {} \t Test last step loss mean {}, std {} ".format(states, actions,
                                                                                     np.mean(
                                                                                         np.array(test_last_losses)),
                                                                                     np.std(
                                                                                         np.array(test_last_losses))))
        print("States {}, actions {} \t Test last step acc mean {}, std {} ".format(states, actions,
                                                                                    np.mean(np.array(test_last_accs)),
                                                                                    np.std(np.array(test_last_accs))))
        results = {
            'losses': test_all_losses,
            'accs': test_all_accs,
            'gt_losses':all_gt_losses,
            'gt_accs': all_gt_accs
        }
        pickle.dump(results, open('results_states_' + str(states) + '_actions_' + str(actions) + '.p', 'wb'))
