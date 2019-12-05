import src.distributions_of_inputs as distributions
import numpy as np


def test_tcre_normal_distribution():
    normal_mean = (0.8 + 2.5) / 2
    normal_sd = (2.5 - 0.8) / 2
    n_return = 1000000
    distn = "normal"
    returned = distributions.tcre_distribution(
        normal_mean, normal_sd, n_return=n_return, tcre_dist=distn,
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert abs(sum(returned > normal_mean) / n_return - 0.5) < 0.01, \
        "This test for skewdness can statistically fail but suggests that the distribution is wrong"


def test_tcre_lognormal_distribution():
    normal_mean = (0.8 + 2.5) / 2
    normal_sd = (2.5 - 0.8) / 2
    n_return = 1000000
    distn = "lognormal"
    returned = distributions.tcre_distribution(
        normal_mean, normal_sd, n_return=n_return, tcre_dist=distn,
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert sum(returned < 0) == 0
