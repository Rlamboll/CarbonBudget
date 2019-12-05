import src.distributions_of_inputs as distributions
import numpy as np
import pytest

def test_tcre_bad_limits():
    low = 5.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    distn = "normal"
    with pytest.raises(AssertionError, match="High and low limits are the wrong way around"):
        returned = distributions.tcre_distribution(
            low, high, likelihood, n_return=1, tcre_dist=distn,
        )


def test_tcre_normal_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    normal_mean = (low + high) / 2
    normal_sd = (high - low) / 2
    n_return = 1000000
    distn = "normal"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn,
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert abs(sum(returned > normal_mean) / n_return - 0.5) < 0.01
    likely_fraction = 1 - (sum(low > returned) + sum(high < returned)) / len(returned)
    assert abs(likely_fraction - likelihood) < 0.01


def test_tcre_lognormal_mean_match_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    normal_mean = (0.8 + 2.5) / 2
    normal_sd = (2.5 - 0.8) / 2
    n_return = 1000000
    distn = "lognormal mean match"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn,
    )
    assert abs(np.mean(returned) - normal_mean) < 0.005
    assert abs(np.std(returned) - normal_sd) < 0.005
    assert sum(returned < 0) == 0
    likely_fraction = 1 - (sum(0.8 > returned) + sum(2.5 < returned)) / len(returned)
    assert likely_fraction > likelihood


def test_tcre_lognormal_pde_distribution():
    low = 0.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    # The median value is the geometric mean
    expected_median = (0.8 * 2.5)**0.5
    n_return = 1000000
    distn = "lognormal"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn,
    )
    assert abs(np.median(returned) - expected_median) < 0.005
    assert sum(returned < 0) == 0
    likely_fraction = 1 - (sum(0.8 > returned) + sum(2.5 < returned)) / len(returned)
    assert abs(likely_fraction - likelihood) < 0.01
