# this file is part of the Selection Function project

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
rc('text', usetex=True)
import numpy as np
from scipy.special import erf
import pylab as plt

def normal(x, m, invvar):
    return np.sqrt(invvar / (2. * np.pi)) * np.exp(-0.5 * invvar * (x - m) ** 2)

def step(x, x0):
    return (x > x0).astype(float)

def likelihood(xobs, xtrue, invvar, xmin):
    Q = 0.5 + 0.5 * erf(np.sqrt(0.5 * invvar) * (xtrue - xmin))
    return normal(xobs, xtrue, invvar) * step(xobs, xmin) / Q

def plot_one_example(xs, ps, x_min, label):
    plt.plot(xs, ps, "k-")
    x1, x2 = np.round(np.min(xs)), np.round(np.max(xs))
    y1, y2 = -0.1 * np.max(ps), 1.1 * np.max(ps)
    plt.fill([x1, x_min, x_min, x1, x1], [y1, y1, y2, y2, y1], 'k', alpha=0.2)
    plt.text(x1 + 0.02 * (x2 - x1), 0.05 * (y2 - y1), label, color='r')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    return None

def main():
    x_min = 6.
    x_range = 16.
    invvar = 1.
    x_trues = np.array([1., 3., 5., 7., 9., 11.])
    dx = 1. / 512.
    x_obss = np.arange(0. + 0.5 * dx, x_range, dx)
    plt.clf()
    for i, x_true in enumerate(x_trues):
        ps = likelihood(x_obss, x_true, invvar, x_min)
        plt.subplot(3, 2, i + 1)
        plot_one_example(x_obss, ps, x_min, r"$x_{\mathrm{true}} = %.2f$" % x_true)
        if i > 3:
            plt.xlabel(r"$x_{\mathrm{obs}}$")
    plt.savefig("p_vs_x_obs.png")
    x_obss = x_min + np.array([0.1, 0.5, 1., 2., 4., 8.])
    x_trues = np.arange(0. + 0.5 * dx, 16., dx)
    plt.clf()
    for i, x_obs in enumerate(x_obss):
        ps = likelihood(x_obs, x_trues, invvar, x_min)
        plt.subplot(3, 2, i + 1)
        plot_one_example(x_trues, ps, x_min, r"$x_{\mathrm{obs}} = %.2f$" % x_obs)
        if i > 3:
            plt.xlabel(r"$x_{\mathrm{true}}$")
    plt.savefig("p_vs_x_true.png")
    return None

if __name__ == "__main__":
    main()
