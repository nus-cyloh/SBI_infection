import numpy as np
from simulator import simulate

SEED = 3247 # for reproducibility

def ppc(beta, gamma, rho, size):
    n = len(beta)
    generator = np.random.default_rng(SEED)
    indices = generator.choice(n, size=size, replace=False)

    pInf, pRew, pDeg = [], [], []
    for i in indices:
        inf, rew, deg = simulate(
            beta[i], gamma[i], rho[i], rng=generator)
        pInf.append(inf)
        pRew.append(rew)
        pDeg.append(deg)
    return (np.array(pInf), np.array(pRew), np.array(pDeg))
