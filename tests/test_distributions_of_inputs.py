import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as calc
import numpy as np
import pandas as pd
import pytest


def test_tcre_bad_limits():
    low = 5.8
    high = 2.5
    likelihood = 0.6827  # we use the 1-sigma range
    distn = "normal"
    with pytest.raises(
        AssertionError, match="High and low limits are the wrong way around"
    ):
        returned = distributions.tcre_distribution(
            low, high, likelihood, n_return=1, tcre_dist=distn
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
        low, high, likelihood, n_return=n_return, tcre_dist=distn
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
        low, high, likelihood, n_return=n_return, tcre_dist=distn
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
    expected_median = (0.8 * 2.5) ** 0.5
    n_return = 1000000
    distn = "lognormal"
    returned = distributions.tcre_distribution(
        low, high, likelihood, n_return=n_return, tcre_dist=distn
    )
    assert abs(np.median(returned) - expected_median) < 0.005
    assert sum(returned < 0) == 0
    likely_fraction = 1 - (sum(0.8 > returned) + sum(2.5 < returned)) / len(returned)
    assert abs(likely_fraction - likelihood) < 0.01


def test_establish_median_temp_dep():
    xy_df = pd.DataFrame(
        {"x": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], "y": [0.0, 0.1, 0.2, 0.2, 0.3, 0.4]}
    )
    quantiles_to_plot = [0.5]
    temp = np.array([2, 3])
    relations = calc.quantile_regression_find_relationships(xy_df, quantiles_to_plot)
    returned = distributions.establish_median_temp_dep(relations, temp)
    # The above data has a 1:1 relationship, so we expect to receive the temp back again
    assert all(abs(x - y) < 1e-14 for x, y in zip(returned, temp))


def test_establish_median_temp_dep_not_skewed():
    # This test is largely equivalent to the above, but includes some very large values
    # that should not impact the overall results.
    xy_df = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "y": [0.0, 0.1, 0.2, 1000, 1000, 1000, 0.1, 0.2, 0.3],
        }
    )
    quantiles_to_plot = [0.5]
    temp = np.array([2, 3])
    relations = calc.quantile_regression_find_relationships(xy_df, quantiles_to_plot)
    returned = distributions.establish_median_temp_dep(relations, temp)
    # The above data has a 1:1 relationship, so we expect to receive the temp back again
    assert all(abs(x - y) < 1e-10 for x, y in zip(returned, temp))
