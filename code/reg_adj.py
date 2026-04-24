"""
Using regression adjustment to further adjust the posterior
bias
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from posterior_predictor import ppc
from helper import culProc, regAdj, plotter1, plotter2

# constants
LIMIT = 0.05 # so 500 elements
pSAM = 40
NAME = "reg_adj"

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
def max_I(data):
    return np.max(data, axis = 1)

def argMax_I(data):
    return np.argmax(data, axis = 1)

def change_max(data):
    n = np.arange(data.shape[0])
    t = np.argmax(data, axis = 1)
    return data[n, t+1]/data[n, t]

def max_rew(data):
    return np.max(data, axis = 1)

def mean_deg(data):
    lg = np.repeat(np.arange(data.shape[1]),
                   data.shape[0]).reshape(data.shape)
    return np.mean(lg * data, axis = 1)

def var_deg(data):
    lg = np.repeat(np.arange(data.shape[1]),
                   data.shape[0]).reshape(data.shape)
    d1 = np.mean(lg * data, axis = 1)
    d2 = np.mean(lg * lg * data, axis = 1)
    return d2 - d1 * d1

def dist(sim_vect, obs_vect):
    return (sim_vect - np.mean(obs_vect))/np.std(sim_vect)

def kern(u):
    return sp.norm.pdf(u)

# compute summary stats
maxI_obs = max_I(obs_inf)
argI_obs = argMax_I(obs_inf)
change_obs = change_max(obs_inf)
maxRew_obs = max_rew(obs_rew)
meanDeg_obs = mean_deg(obs_deg)
varDeg_obs = var_deg(obs_deg)

maxI_sim = max_I(sim_inf)
argI_sim = argMax_I(sim_inf)
change_sim = change_max(sim_inf)
maxRew_sim = max_rew(sim_rew)
meanDeg_sim = mean_deg(sim_deg)
varDeg_sim = var_deg(sim_deg)

# scale data and compute dist
distMaxI = dist(maxI_sim, maxI_obs)
distArgI = dist(argI_sim, argI_obs)
distChange = dist(change_sim, change_obs)
distMaxRew = dist(maxRew_sim, maxRew_obs)
distMeanDeg = dist(meanDeg_sim, meanDeg_obs)
distVarDeg = dist(varDeg_sim, varDeg_obs)
d = np.sqrt(distMaxI**2 + distArgI**2 + distChange**2 +
            distMaxRew**2 + distMeanDeg**2 + distVarDeg ** 2)

# accept only top few data
bound = np.quantile(d, LIMIT)
ind = d <= bound
sel_beta = betas[ind]
sel_gamma = gammas[ind]
sel_rho = rhos[ind]

# regression
sim_sumStats = np.column_stack([
        maxI_sim, argI_sim, change_sim, 
        maxRew_sim, meanDeg_sim, varDeg_sim
        ])
sim_accStats = sim_sumStats[ind]
obs_sumStats = np.mean(np.array([
    maxI_obs, argI_obs, change_obs,
    maxRew_obs, meanDeg_obs, varDeg_obs
    ]), axis = 1).reshape(1, -1)

adj_beta = regAdj(sel_beta, sim_accStats, obs_sumStats, kern)
adj_gamma = regAdj(sel_gamma, sim_accStats, obs_sumStats, kern)
adj_rho = regAdj(sel_rho, sim_accStats, obs_sumStats, kern)

# ppc
sim_tuple = ppc(adj_beta, adj_gamma, adj_rho, pSAM)
obs_tuple = culProc(obs_inf, obs_rew, obs_deg)

# plots
plotter1(adj_beta, adj_gamma, adj_rho, NAME)
plotter2(sim_tuple, obs_tuple, NAME)

