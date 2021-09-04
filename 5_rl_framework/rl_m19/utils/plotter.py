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
        config: {
            id: unique identifier,
            title: '',
            xlabel: '',
            ylabel: [''],
            x_data: [[]],
            y_data: [[]]
        }
        '''
        fig = plt.figure(config['id'])
        axes = fig.get_axes()
        if len(axes) == 0:
            plt.title(config['title'])
            plt.xlabel(config['xlabel'])
            for i in range(len(config['ylabel'])):
                if i == 0:
                    plt.plot(
                        config['x_data'][i],
                        config['y_data'][i],
                        label=config['ylabel'][i])
                    axes = fig.get_axes()
                    ax = axes[0]
                else:
                    twin = ax.twinx()
                    if i >= 2:
                        fig.subplots_adjust(right=1.0 - 0.25*(i-1))
                        twin.spines.right.set_position(("axes", 1.0 + 0.2*(i-1)))
                    twin.plot(
                        config['x_data'][i],
                        config['y_data'][i],
                        label=config['ylabel'][i])
        else:
            for i in range(len(config['ylabel'])):
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
        return axes

    def plot_scatter(self, config):
        '''
        config: {
            axes: plt.Axes,
            x_data: [],
            y_data: [],
            s: int, e.g. 25,
            c: 'red', 'green', etc.,
            marker: '^', 'v', 'o'
        }
        '''
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
        # https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.PathCollection
        fig = plt.figure(config['id'])
        path_collection = config['axes'].scatter(
            x=config['x_data'],
            y=config['y_data'],
            s=config['s'],
            c=config['c'],
            marker=config['marker'])
        if is_ipython:
            display.clear_output(wait=True)
            display.display(fig)
        else:
            plt.pause(0.1)  # pause a bit so that plots are updated
        return path_collection

    def plot_end(self):
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    p = Plotter()
    line = p.plot_single({
        'id': 1,
        'title': 'single_line',
        'xlabel': 't',
        'ylabel': 'l1',
        'x_data': range(4),
        'y_data': [1, 2, 3, 4]
    })
    plt.pause(1)
    p.plot_single({
        'id': 1,
        'title': 'single_line',
        'xlabel': 't',
        'ylabel': 'l1',
        'x_data': range(5),
        'y_data': [1, 2, 3, 5, 8],
    })

    p.plot_single_with_mean({
        'id': 2,
        'title': 'single_with_mean',
        'xlabel': 't',
        'ylabel': 'l1',
        'x_data': range(4),
        'y_data': [1, 2, 3, 4],
        "m": 3
    })
    plt.pause(1)
    p.plot_single_with_mean({
        'id': 2,
        'title': 'single_with_mean',
        'xlabel': 't',
        'ylabel': 'l1',
        'x_data': range(5),
        'y_data': [1, 2, 3, 5, 8],
        "m": 3
    })

    p.plot_multiple({
        'id': 3,
        'title': 'multiple_lines',
        'xlabel': 't',
        'ylabel': ['l1', 'l2'],
        'x_data': [range(4), range(5)],
        'y_data': [[1, 2, 3, 4], [1, 2, 3, 5, 8]],
    })
    plt.pause(1)
    p.plot_multiple({
        'id': 3,
        'title': 'multiple_lines',
        'xlabel': 't',
        'ylabel': ['l1', 'l2'],
        'x_data': [range(5), range(6)],
        'y_data': [[1, 2, 3, 4, 5], [1, 2, 3, 5, 8, 13]],
    })
    plt.show(block=True)
