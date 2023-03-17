import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

def plot_trained_gp(targets, means, preds, fig_count=0, show=False):
    lower, upper = preds.confidence_region()
    fig_count += 1
    plt.figure(fig_count)
    plt.fill_between(list(range(lower.shape[0])), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
    plt.plot(means.squeeze(), 'r', label='GP Mean')
    plt.plot(targets.squeeze(), '*k', label='Targets')
    plt.legend()
    plt.title('Fitted GP')
    plt.xlabel('Time (s)')
    plt.ylabel('v')
    if show:
        plt.show()

    return fig_count
