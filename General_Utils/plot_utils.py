import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot(save_fp, x_vals, y_vals, title, x_label, y_label):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # make sure the x axis uses integers rather than floats
    title = title.replace("_", " ")
    plt.title(title)
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_fp)
    plt.clf()
    plt.close()


def plot_histogram(vals_, title, y_label, x_label, save_fp,
                   n_bins=10):
    vals = np.array(vals_, dtype=float)
    plt.hist(vals, density=False, bins=n_bins)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(save_fp)
    plt.clf()
    plt.close()
