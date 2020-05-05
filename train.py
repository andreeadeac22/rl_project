import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter

from models import *
from dataset import *


def loss_fn(output, target, reduction='mean'):
    loss = (target.float().squeeze() - output.squeeze()) ** 2
    return loss.sum() if reduction == 'sum' else loss.mean()


def train(data):
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0
    for i in range(len(data)):
        data[i] = data[i].to(args.device)
    node_feat, adj_mat, adj_mask, vs = data
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]

    for step in range(iteration_steps-1):
        optimizer.zero_grad()
        output = model((node_feat[step], adj_mat, adj_mask))
        loss = loss_fn(output, vs[step + 1])
        #print("output ", output)
        #print("vs[step + 1] ", vs[step + 1])
        #print("loss ", loss)
        loss.backward()
        optimizer.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)

    print('Train Epoch: {}, samples: {} \tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
        epoch + 1, n_samples, loss.item(), train_loss / n_samples,
        time_iter))


def test(data):
    model.eval()
    start = time.time()
    test_loss, correct, n_samples = 0, 0, 0
    for i in range(len(data)):
        data[i] = data[i].to(args.device)

    node_feat, adj_mat, adj_mask, vs = data
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]
    for step in range(iteration_steps - 1):
        output = model((node_feat[step], adj_mat, adj_mask))
        loss = loss_fn(output, vs[step + 1])
        test_loss += loss.item()
        n_samples += len(output)

    print('Test set (epoch {}): Average loss: {:.6f} \n'.format(
        epoch + 1,
        test_loss / n_samples))
    return test_loss / n_samples


parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('--train_num_states', type=int, default=20, help='number of states')
parser.add_argument('--train_num_actions', type=int, default=5, help='number of actions')

parser.add_argument('--test_num_states', type=int, default=100, help='number of states')
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

iterable_train_dataset = GraphData(num_states=args.train_num_states, num_actions=args.train_num_actions, epsilon=args.epsilon)
train_loader = torch.utils.data.DataLoader(iterable_train_dataset, batch_size=None)


iterable_test_dataset = GraphData(num_states=args.test_num_states, num_actions=args.test_num_actions, epsilon=args.epsilon)
test_loader = torch.utils.data.DataLoader(iterable_test_dataset, batch_size=None)

for epoch in range(args.num_train_graphs):
    train(next(iter(train_loader)))

test_losses = []
for epoch in range(args.num_test_graphs):
    test_losses += [test(next(iter(test_loader)))]
print("Test loss mean {}, std {} ".format(np.mean(np.array(test_losses)), np.std(np.array(test_losses))))



