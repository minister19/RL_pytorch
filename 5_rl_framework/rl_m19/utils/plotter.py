import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Plotter():
    def __init__(self):
        plt.ion()  # set up matplotlib

    def plot_single(self, config):
        '''
        config: {
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabel: '',
            x_data: [],
            y_data: []
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            plt.plot(config['x_data'], config['y_data'], label=config['ylabel'])
        else:
            ax = axes[0]
            line, = ax.get_lines()
            line.set_xdata(config['x_data'])
            line.set_ydata(config['y_data'])
            ax.relim()
            ax.autoscale_view(True, True, True)
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
            x_data: [],
            y_data: [],
            m: int
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        _data = config['y_data']
        m = config['m']
        if m > 0 and len(_data) > m:
            __data = torch.tensor(_data, dtype=torch.float)
            mean = __data.unfold(0, m, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(m - 1), mean)).numpy()
        else:
            means = [None] * len(_data)
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            plt.plot(config['x_data'], config['y_data'], label=config['ylabel'])
            plt.plot(config['x_data'], means, label=config['ylabel'] + '_mean')
        else:
            ax = axes[0]
            line, meanline = ax.get_lines()
            line.set_xdata(config['x_data'])
            line.set_ydata(config['y_data'])
            meanline.set_xdata(config['x_data'])
            meanline.set_ydata(means)
            ax.relim()
            ax.autoscale_view(True, True, True)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(fig)
        else:
            plt.pause(0.1)  # pause a bit so that plots are updated

    def plot_multiple(self, config):
        '''
        configs: {
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabels: [''],
            x_data: [[]],
            y_data: [[]]
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            for i in range(len(config['ylabels'])):
                if i == 0:
                    plt.plot(config['x_data'][i], config['y_data'][i], label=config['ylabels'][i])
                    axes = fig.get_axes()
                    ax = axes[0]
                else:
                    twin = ax.twinx()
                    # twin.spines.right.set_position(("axes", 1.2))
                    twin.plot(config['x_data'][i], config['y_data'][i], label=config['ylabels'][i])
        else:
            for i in range(len(config['ylabels'])):
                ax = axes[i]
                line, = ax.get_lines()
                line.set_xdata(config['x_data'][i])
                line.set_ydata(config['y_data'][i])
                ax.relim()
                ax.autoscale_view(True, True, True)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(fig)
        else:
            plt.pause(0.1)  # pause a bit so that plots are updated

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
