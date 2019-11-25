import src.distributions_of_inputs as distributions
import numpy as np

def test_tcre_distribution():
    normal_mean = (0.8 + 2.5) / 2
    normal_sd = (2.5 - 0.8) / 2
    n_return = 1000000
    returned = distributions.tcre_distribution(normal_mean, normal_sd, n_return=n_return)
    assert abs(np.mean(returned)-normal_mean) < 0.005
    assert abs(np.std(returned)-normal_sd) < 0.005
