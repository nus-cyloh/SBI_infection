"""
Helper functions for various utilities
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# numerics
def rVal(x, y):
    return np.corrcoef(x, y)[1,0].item()

def rDisp(x, y):
    return "R^2: {:3f}".format(rVal(x, y))

def culProc(inf, rew, deg):
    tInf = np.mean(np.transpose(inf, (1,0)), axis = 1)
    tRew = np.mean(np.transpose(rew, (1,0)), axis = 1)
    tDeg = np.mean(np.transpose(deg, (1,0)), axis = 1)
    return (tInf, tRew, tDeg)

def regAdj(param, acc, obs, kern_func):
    obs_mean = np.mean(obs, axis = 0)
    d = np.linalg.norm(acc - obs_mean, axis = 1)
    h = np.max(d)
    if h == 0: return param
    u = d / h
    w = kern_func(u)
    model = LinearRegression()
    model.fit(acc, param, sample_weight = w)
    b = model.coef_
    return param - (acc - obs_mean) @ b

def like_clean(like, beta, gamma, rho):
    bound = np.quantile(like, 0.95)
    ind = like >= bound
    sel_beta = beta[ind]
    sel_gamma = gamma[ind]
    sel_rho = rho[ind]
    sel_like = like[ind]
    return (np.exp(sel_like), sel_beta, sel_gamma, sel_rho)

# plots
def post_plot(ax, vect, name):
    ax.set_title(f"{name} post")
    ax.hist(vect, bins = 50, color="c", alpha=0.8, density = True)
    ax.axvline(x=np.mean(vect), color="b")

def pair_plot(ax, x, y, x_title, y_title):
    ax.set_title(f"{y_title} v {x_title}")
    ax.scatter(x, y, color="c")
    ax.set(xlabel = x_title, ylabel = y_title)
    ax.annotate(rDisp(x, y), (0, 1), xycoords = "axes fraction")

def ppc_plot(ax, sim, obs, title, x_label, y_label):
    for i in sim:
        ax.plot(i, color="c", alpha=0.15)
    ax.plot(obs, color="b", linewidth=2.5, label = "obs")
    ax.set_title(title)
    ax.set(xlabel = x_label, ylabel = y_label)
    ax.legend()

def post_line(ax, x, y, name):
    newX = x[np.argsort(x)]
    newY = y[np.argsort(x)]
    ax.set_title(f"{name} post")
    ax.plot(newX, newY, color="c")
    ax.set(xlabel = name)
    ax.axvline(x = np.mean(x), color="b")

def plotter1(beta, gamma, rho, typ):
    f, ax = plt.subplots(2, 3, figsize = (15, 10))
    plt.tight_layout(pad=2.0)
    post_plot(ax[0, 0], beta, "beta")
    post_plot(ax[0, 1], gamma, "gamma")
    post_plot(ax[0, 2], rho, "rho")
    pair_plot(ax[1, 0], beta, gamma, "beta", "gamma")
    pair_plot(ax[1, 1], gamma, rho, "gamma", "rho")
    pair_plot(ax[1, 2], rho, beta, "rho", "beta")
    plt.savefig(f"../img/{typ}_posteriors.png")

def plotter2(sim, obs, typ):
    f, ax = plt.subplots(1, 3, figsize = (15, 5))
    plt.tight_layout(pad=2.0)
    ppc_plot(ax[0], sim[0], obs[0], "Infected proportion",
             "time", "proportion")
    ppc_plot(ax[1], sim[1][:,:61], obs[1][:61], "Rewiring count",
             "time", "count/step")
    ppc_plot(ax[2], sim[2], obs[2], "Final degree",
             "degree", "frequency")
    plt.savefig(f"../img/{typ}_ppc.png")

def plotter3(loglike, beta, gamma, rho, typ):
    stats = like_clean(loglike, beta, gamma, rho) 
    f, ax = plt.subplots(2, 3, figsize = (15, 10))
    post_line(ax[0, 0], stats[1], stats[0], "beta")
    post_line(ax[0, 1], stats[2], stats[0], "gamma")
    post_line(ax[0, 2], stats[3], stats[0], "rho")
    pair_plot(ax[1, 0], stats[1], stats[2], "beta", "gamma")
    pair_plot(ax[1, 1], stats[2], stats[3], "gamma", "rho")
    pair_plot(ax[1, 2], stats[3], stats[1], "rho", "beta")
    plt.savefig(f"../img/{typ}_posteriors.png")
