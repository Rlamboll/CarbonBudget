import numpy as np


def tcre_distribution(mean, sd, n_return=1):
    return np.random.normal(mean, sd, n_return)
