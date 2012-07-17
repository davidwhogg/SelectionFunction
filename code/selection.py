# this file is part of the Selection Function project

import matplotlib
matplotlib.use("Agg")
from matplotlib import rc
rc("font",**{"family":"serif","serif":"Computer Modern Roman","size":12})
rc("text", usetex=True)
import numpy as np
from scipy.special import erf
import pylab as plt

suffix = ".pdf"

def normal(x, m, invvar):
    return np.sqrt(invvar / (2. * np.pi)) * np.exp(-0.5 * invvar * (x - m) ** 2)

def step(x, x0):
    return (x > x0).astype(float)

def likelihood(xobs, xtrue, invvar, xmin):
    # Q = 0.5 + 0.5 * erf(np.sqrt(0.5 * invvar) * (xtrue - xmin))
    Q = 1.
    return normal(xobs, xtrue, invvar) * step(xobs, xmin) / Q

def prior(xtrue):
    xtrue = np.atleast_1d(xtrue)
    xmin = 1.
    xmax = 32.
    alpha = -2.35
    ap1 = alpha + 1.
    Q = 1. / (xmax ** ap1 / ap1 - xmin ** ap1 / ap1)
    p = np.zeros_like(xtrue)
    okay = (xtrue > xmin) * (xtrue < xmax)
    if np.sum(okay) > 0:
        p[okay] = Q * xtrue[okay] ** alpha
    print alpha, ap1, Q, np.min(p), np.max(p)
    return p

def plot_one_example(ax, xs, ps, x_min, label, semilogy, yplotmax):
    print ax, semilogy
    x1, x2 = np.round(np.min(xs)), np.round(np.max(xs))
    y1, y2 = -0.1 * yplotmax, 1.1 * yplotmax
    if semilogy:
        y2 = 2. * yplotmax
        y1 = 0.001 * y2
        y3 = 0.1 * y1
        ps[ps < y3] = y3
    print y1, y2
    ax.plot(xs, ps, "k-")
    ax.fill([x1, x_min, x_min, x1, x1], [y1, y1, y2, y2, y1], "k", alpha=0.2)
    ytext = 0.05 * (y2 - y1)
    if semilogy:
        ax.semilogy()
        ytext = np.exp(np.log(y1) + 0.05 * (np.log(y2) - np.log(y1)))
    ax.text(x1 + 0.04 * (x2 - x1), ytext, label, color="r")
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    return None

def main():
    x_min = 6.
    x_range = 16.
    invvar = 1.
    dx = 1. / 512.
    x_obss = np.arange(0. + 0.5 * dx, x_range, dx)
    x_trues = np.array([1., 3., 5., 7., 9., 11.])
    fig1, fig2 = plt.figure(), plt.figure()
    for i, x_true in enumerate(x_trues):
        ps = likelihood(x_obss, x_true, invvar, x_min)
        ax1, ax2 = [fig.add_subplot(3, 2, i + 1) for fig in [fig1, fig2]]
        [plot_one_example(ax, x_obss, ps, x_min, r"$x_{\mathrm{true}} = %.2f$" % x_true, b, 0.4)
         for ax, b in [(ax1, False), (ax2, True)]]
        if i == 2:
            [ax.set_ylabel(r"$p(x_{\mathrm{obs}},{\mathrm{in}}\,|\,x_{\mathrm{true}})$") for ax in [ax1, ax2]]
        if i > 3:
            [ax.set_xlabel(r"$x_{\mathrm{obs}}$") for ax in [ax1, ax2]]
    fig1.savefig("p_vs_x_obs"+suffix), fig2.savefig("p_vs_x_obs_log"+suffix)
    x_obss = x_min + np.array([0.1, 0.5, 1., 2., 4., 8.])
    x_trues = np.arange(0. + 0.5 * dx, x_range, dx)
    fig1, fig2 = plt.figure(), plt.figure()
    for i, x_obs in enumerate(x_obss):
        ps = likelihood(x_obs, x_trues, invvar, x_min)
        ax1, ax2 = [fig.add_subplot(3, 2, i + 1) for fig in [fig1, fig2]]
        [plot_one_example(ax, x_trues, ps, x_min, r"$x_{\mathrm{obs}} = %.2f$" % x_obs, b, 0.4)
         for ax, b in [(ax1, False), (ax2, True)]]
        if i == 2:
            [ax.set_ylabel(r"$p(x_{\mathrm{obs}},{\mathrm{in}}\,|\,x_{\mathrm{true}})$") for ax in [ax1, ax2]]
        if i > 3:
            [ax.set_xlabel(r"$x_{\mathrm{true}}$") for ax in [ax1, ax2]]
    fig1.savefig("likelihood"+suffix), fig2.savefig("likelihood_log"+suffix)
    x_obss = x_min + np.array([0.1, 0.5, 1., 2., 4., 8.])
    x_trues = np.arange(0. + 0.5 * dx, x_range, dx)
    fig1, fig2 = plt.figure(), plt.figure()
    ps = prior(x_trues)
    for i in range(6): 
        ax1, ax2 = [fig.add_subplot(3, 2, i + 1) for fig in [fig1, fig2]]
        [plot_one_example(ax, x_trues, ps, x_min, "", b, 1.35)
         for ax, b in [(ax1, False), (ax2, True)]]
        if i == 2:
            [ax.set_ylabel(r"$p(x_{\mathrm{true}})$") for ax in [ax1, ax2]]
        if i > 3:
            [ax.set_xlabel(r"$x_{\mathrm{true}}$") for ax in [ax1, ax2]]
    fig1.savefig("prior"+suffix), fig2.savefig("prior_log"+suffix)
    x_obss = x_min + np.array([0.1, 0.5, 1., 2., 4., 8.])
    x_trues = np.arange(0. + 0.5 * dx, x_range, dx)
    fig1, fig2 = plt.figure(), plt.figure()
    for i, x_obs in enumerate(x_obss):
        ps = prior(x_trues) * likelihood(x_obs, x_trues, invvar, x_min)
        ps /= np.sum(ps * dx)
        ax1, ax2 = [fig.add_subplot(3, 2, i + 1) for fig in [fig1, fig2]]
        [plot_one_example(ax, x_trues, ps, x_min, r"$x_{\mathrm{obs}} = %.2f$" % x_obs, b, 0.4)
         for ax, b in [(ax1, False), (ax2, True)]]
        if i == 2:
            [ax.set_ylabel(r"$p(x_{\mathrm{true}}\,|\,x_{\mathrm{obs}},{\mathrm{in}})$") for ax in [ax1, ax2]]
        if i > 3:
            [ax.set_xlabel(r"$x_{\mathrm{true}}$") for ax in [ax1, ax2]]
    fig1.savefig("posterior"+suffix), fig2.savefig("posterior_log"+suffix)
    x_obss = np.arange(0. + 0.5 * dx, x_range, dx)
    x_trues = np.arange(0. + 0.5 * dx, x_range, dx)
    fig1, fig2 = plt.figure(), plt.figure()
    ps = np.zeros_like(x_obss)
    for x_true in x_trues:
        print x_true
        ps += prior(x_true) * likelihood(x_obss, x_true, invvar, x_min) * dx
    for i in range(6):
        ax1, ax2 = [fig.add_subplot(3, 2, i + 1) for fig in [fig1, fig2]]
        [plot_one_example(ax, x_obss, ps, x_min, "", b, 1.35)
         for ax, b in [(ax1, False), (ax2, True)]]
        if i == 2:
            [ax.set_ylabel(r"$p(x_{\mathrm{obs}},\mathrm{in})$") for ax in [ax1, ax2]]
        if i > 3:
            [ax.set_xlabel(r"$x_{\mathrm{true}}$") for ax in [ax1, ax2]]
    fig1.savefig("prior_obs"+suffix), fig2.savefig("prior_obs_log"+suffix)
    return None

if __name__ == "__main__":
    main()
