"""
Naive ABC rejection algorithm with initial summary statistic
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator import simulate
from posterior_predictor import ppc
from helper import culProc, plotter1, plotter2

# constants
LIMIT = 0.05 # so 500 elements
pSAM = 40 # no of samples for overlay
NAME = "naive_abc"

# load data
obs_inf = np.load("../data_obs/infected.npy")
obs_rew = np.load("../data_obs/rewired.npy")
obs_deg = np.load("../data_obs/degrees.npy")

sim_inf = np.load("../data_sim/infected_sim.npy")
sim_rew = np.load("../data_sim/rewire_sim.npy")
sim_deg = np.load("../data_sim/degree_sim.npy")

betas = np.load("../data_sim/beta.npy")
gammas = np.load("../data_sim/gamma.npy")
rhos = np.load("../data_sim/rho.npy")

# helper functions for summary statistics
# assumes data sets follows structure of rep_id -> values
print(obs_deg[0])
def max_I(data):
    return np.max(data, axis = 1)

def change_max(data):
    n = np.arange(data.shape[0])
    t = np.argmax(data, axis = 1)
    return data[n, t+1]/data[n, t]

def max_rew(data):
    return np.max(data, axis = 1)

def dist(sim_vect, obs_vect):
    return (sim_vect - np.mean(obs_vect))/np.std(sim_vect)

# compute summary stats
maxI_obs = max_I(obs_inf)
change_obs = change_max(obs_inf)
maxRew_obs = max_rew(obs_rew)

maxI_sim = max_I(sim_inf)
change_sim = change_max(sim_inf)
maxRew_sim = max_rew(sim_rew)

# scale data and compute dist
distMaxI = dist(maxI_sim, maxI_obs)
distChange = dist(change_sim, change_obs)
distMaxRew = dist(maxRew_sim, maxRew_obs)
d = np.sqrt(distMaxI**2 + distChange**2 + distMaxRew**2)

# accept only top few data
bound = np.quantile(d, LIMIT)
ind = d <= bound
sel_beta = betas[ind]
sel_gamma = gammas[ind]
sel_rho = rhos[ind]

# ppc
sim_tuple = ppc(sel_beta, sel_gamma, sel_rho, pSAM)
obs_tuple = culProc(obs_inf, obs_rew, obs_deg)

# plots
plotter1(sel_beta, sel_gamma, sel_rho, NAME)
plotter2(sim_tuple, obs_tuple, NAME)
