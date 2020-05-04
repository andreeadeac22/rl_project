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


def train(train_loader):
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0
    for batch_idx, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(args.device)
        node_feat, adj_mat, vs = data
        # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
        # adj_mat.shape: a, s, s, 2 (p, gamma)
        iteration_steps = node_feat.shape[0]

        for step in range(iteration_steps-1):
            optimizer.zero_grad()
            output = model((node_feat[step], adj_mat))
            loss = loss_fn(output, vs[step + 1])
            #print("output ", output)
            #print("vs[step + 1] ", vs[step + 1])
            #print("loss ", loss)
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)

        if batch_idx % 20 == 0:
            print('Train Epoch: {}, samples: {} \tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                epoch + 1, n_samples, loss.item(), train_loss / n_samples,
                time_iter / (batch_idx + 1)))


def test(test_loader):
    model.eval()
    start = time.time()
    test_loss, correct, n_samples = 0, 0, 0
    for batch_idx, data in enumerate(test_loader):
        for i in range(len(data)):
            data[i] = data[i].to(args.device)

        node_feat, adj_mat, vs = data
        # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
        # adj_mat.shape: a, s, s, 2 (p, gamma)
        iteration_steps = node_feat.shape[0]
        for step in range(iteration_steps - 1):
            output = model((node_feat[step], adj_mat))
            loss = loss_fn(output, vs[step + 1])
            test_loss += loss.item()
            n_samples += len(output)

    print('Test set (epoch {}): Average loss: {:.4f} \n'.format(
        epoch + 1,
        test_loss / n_samples))
    return test_loss / n_samples


parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('--num_states', type=int, default=20, help='number of states')
parser.add_argument('--num_actions', type=int, default=5, help='number of actions')
parser.add_argument('--epsilon', type=float, default=1e-8, help='termination condition (difference between two '
                                                                'consecutive values)')

parser.add_argument('--filters', type=int, default=[64, 64], help='Hidden dim for node')

parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200, help='Max number of epochs')
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

iterable_dataset = GraphData(num_states=args.num_states, num_actions=args.num_actions, epsilon=args.epsilon)
loader = torch.utils.data.DataLoader(iterable_dataset, batch_size=None)

min_loss = float('inf')
wait = 0
for epoch in range(args.epochs):
    train(loader)  # no need to evaluate after each epoch
    loss = test(loader)
    if loss < min_loss:
        print("Better loss")
        min_loss = loss
    else:
        print("Patience ", wait + 1)
        wait += 1
        if wait > args.patience:
            break
