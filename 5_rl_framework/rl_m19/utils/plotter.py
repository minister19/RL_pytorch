import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Plotter():
    def __init__(self):
        plt.ion()  # set up matplotlib

    def plot_line(self, config):
        '''
        config: {
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabel: '',
            data: []
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            plt.ylabel(config['ylabel'])
            plt.plot(config['data'])
        else:
            _data = config['data']
            line, = axes[0].get_lines()
            line.set_xdata(range(len(_data)))
            line.set_ydata(_data)
            axes[0].relim()
            axes[0].autoscale_view(True, True, True)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(fig)
        else:
            plt.pause(0.1)  # pause a bit so that plots are updated

    def plot_single_with_mean(self, config):
        '''
        config: {
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabel: '',
            data: [],
            m: int
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        _data = config['data']
        m = config['m']
        if m > 0 and len(_data) > m:
            __data = torch.tensor(_data, dtype=torch.float)
            mean = __data.unfold(0, m, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(m - 1), mean)).numpy()
        else:
            means = []
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            plt.ylabel(config['ylabel'])
            plt.plot(_data)
            plt.plot([])
        else:
            line, meanline = axes[0].get_lines()
            line.set_xdata(range(len(_data)))
            line.set_ydata(_data)
            meanline.set_xdata(range(len(means)))
            meanline.set_ydata(means)
            axes[0].relim()
            axes[0].autoscale_view(True, True, True)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(fig)
        else:
            plt.pause(0.1)  # pause a bit so that plots are updated

    def plot_multiple_line(self, configs):
        '''
        configs: [{
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabel: '',
            data: []
        }]
        '''
        pass

    def plot_end(self):
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    p = Plotter()
    line = p.plot_line({
        'id': 1,
        'title': 'episode_t',
        'xlabel': 'iteration',
        'ylabel': 'lifespan',
        'data': [1, 3, 5],
    })
    plt.pause(1)
    p.plot_line({
        'id': 1,
        'title': 'episode_t',
        'xlabel': 'iteration',
        'ylabel': 'lifespan',
        'data': [1, 2, 3, 5, 8],
    })
    plt.show(block=True)
