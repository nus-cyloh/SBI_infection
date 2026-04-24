"""
Convert existing csv files to npy files for ease of usage
"""

import numpy as np
import pandas as pd

# load data to convert
inf = pd.read_csv("../data_obs/infected_timeseries.csv")
rew = pd.read_csv("../data_obs/rewiring_timeseries.csv")
deg = pd.read_csv("../data_obs/final_degree_histograms.csv")

# split by replication id
spl_inf = np.array([g for v, g in inf.groupby('replicate_id')])
spl_rew = np.array([g for v, g in rew.groupby('replicate_id')])
spl_deg = np.array([g for v, g in deg.groupby('replicate_id')])

# transpose to read by col instead -- easier to extract data
# to compare with simulated data
# additionally, remove replication_id and time since can use
# index -- make obs data similar format as sim data
np_inf = np.transpose(spl_inf, (0, 2, 1))[:, 2, :]
np_rew = np.transpose(spl_rew, (0, 2, 1))[:, 2, :]
np_deg = np.transpose(spl_deg, (0, 2, 1))[:, 2, :]

# save as npy since faster
np.save("../data_obs/infected.npy", np_inf)
np.save("../data_obs/rewired.npy", np_rew)
np.save("../data_obs/degrees.npy", np_deg)
