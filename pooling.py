import numpy as np, glob

def pool_mean_std(E):
    mu = E.mean(axis=0)
    sd = E.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype("float32")

