import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Plotter():
    def __init__(self):
        # set up matplotlib
        plt.ion()

    def plot_tensor(self, data, m=0):
        fig1 = plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(data.numpy())
        if m > 0 and len(data) >= m:
            mean = data.unfold(0, m, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(m - 1), mean))
            plt.plot(means.numpy())
        plt.pause(0.005)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_list_ndarray(self, x, m=0):
        x_ = torch.tensor(x, dtype=torch.float)
        self.plot_tensor(x_, m)

    def plot_end(self):
        plt.ioff()
        plt.show()
