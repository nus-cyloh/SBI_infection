"""
Generate large amounts of data from prior to use.
For testing, extract subset to use to reduce time spent.
Note that filtering is a lot cheaper than simulation.
"""

import numpy as np
from simulator import simulate

# constants
Nsim = 10000

# prior data generation
beta = np.random.uniform(0.05, 0.50, Nsim)
gamma = np.random.uniform(0.02, 0.20, Nsim)
rho = np.random.uniform(0.0, 0.8, Nsim)

# save as npy because faster to load; used for future
# filtering
np.save("../data_sim/beta.npy", beta)
np.save("../data_sim/gamma.npy", gamma)
np.save("../data_sim/rho.npy", rho)

# generate data to store based on priors
inf_sim = []
rew_sim = []
deg_sim = []
for i in range(Nsim):
    inf_frac, rew_count, deg_hist = simulate(
            beta[i],gamma[i], rho[i]
            )
    inf_sim.append(inf_frac)
    rew_sim.append(rew_count)
    deg_sim.append(deg_hist)

np_inf = np.array(inf_sim)
np_rew = np.array(rew_sim)
np_deg = np.array(deg_sim)

np.save("../data_sim/infected_sim.npy", np_inf)
np.save("../data_sim/rewire_sim.npy", np_rew)
np.save("../data_sim/degree_sim.npy", np_deg)
