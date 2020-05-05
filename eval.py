import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_test_losses():
    for num_states in [20,50,100]:
        states = pickle.load(open("results" + str(num_states) + ".p", 'rb'))
        losses, accs, gt_accs = states['losses'], states['accs'], states['gt_accs']
        iteration_lengths = [len(graph_losses) for graph_losses in losses]
        print("Minimum ", min(iteration_lengths))
        print("Maximum ", max(iteration_lengths))

        chopped_losses = [graph_losses[:min(iteration_lengths)] for graph_losses in losses]
        chopped_accs = [graph_accs[:min(iteration_lengths)] for graph_accs in accs]
        chopped_gt_accs = [graph_gt_accs[:min(iteration_lengths)] for graph_gt_accs in gt_accs]

        chopped_losses = np.stack(chopped_losses, axis=0)
        chopped_accs = np.stack(chopped_accs, axis=0)
        chopped_gt_accs = np.stack(chopped_gt_accs, axis=0)


        losses_over_iter = np.mean(chopped_losses, axis=0)
        accs_over_iter = np.mean(chopped_accs, axis=0)
        gt_accs_over_iter = np.mean(chopped_gt_accs, axis=0)


        sns.set()

        plt.figure(1)
        plt.xlabel('iteration step')
        plt.ylabel('loss')
        plt.title('Loss over iteration steps')

        plt.figure(2)
        plt.xlabel('iteration step')
        plt.ylabel('policy accuracy')
        plt.title('Policy accuracy over iteration steps')

        plt.figure(1)
        plt.yscale("log")
        plt.plot(range(min(iteration_lengths)), losses_over_iter, label='|s|=' + str(num_states))

        plt.figure(2)
        plt.plot(range(min(iteration_lengths)), accs_over_iter, label='Predicted, |s|=' + str(num_states))
        plt.plot(range(min(iteration_lengths)), gt_accs_over_iter, '--', label='Ground-truth, |s|=' + str(num_states))

        plt.figure(1)
        plt.legend()
        plt.savefig('loss' + str(num_states) + '.jpg')

        plt.figure(2)
        plt.legend()
        plt.savefig('acc' + str(num_states) + '.jpg')


plot_test_losses()