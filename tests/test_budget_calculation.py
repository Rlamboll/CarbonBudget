import re

import pytest

import src.budget_calculator_functions
import numpy as np


def test_calculate_budget():
    dT_target = 2.5
    zec = 0.4
    historical_dT = 1
    non_co2_dT = 0.1
    tcre = 1
    earth_feedback_co2 = 0.1
    budget = src.budget_calculator_functions.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcre, earth_feedback_co2
    )
    expected = 0.9
    assert abs(budget - expected) < 1e-15


def test_calculate_multiple_budgets():
    dT_target = np.array([2.5, 3.5])
    zec = np.array([0.4, 0.45])
    historical_dT = np.array([1, 0.95])
    non_co2_dT = np.array([0.1, 0.2])
    tcre = np.array([1, 0.5])
    earth_feedback_co2 = np.array([0.1, 0.2])
    budget = src.budget_calculator_functions.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcre, earth_feedback_co2
    )
    expected = np.array([0.9, 3.6])
    assert all(abs(budget - expected) < 1e-15)


def test_calculate_multiple_tcres():
    dT_target = 1.5
    zec = 0
    historical_dT = 1.1
    non_co2_dT = 0.108
    tcre = np.random.normal(1.65 / 3664.0, 0.4 / 3664.0, 10000000)
    earth_feedback_co2 = 54
    budget = src.budget_calculator_functions.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcre, earth_feedback_co2
    )
    expected = 594.417
    assert abs(np.median(budget) - expected) < 0.5


def test_mixed_calculation_of_budgets():
    dT_target = np.array([2.5, 3.5])
    zec = 0.4
    historical_dT = np.array([1, 0.95])
    non_co2_dT = np.array([0.1, 0.2])
    tcre = 1
    earth_feedback_co2 = np.array([0.1, 0.2])
    budget = src.budget_calculator_functions.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcre, earth_feedback_co2
    )
    expected = np.array([0.9, 1.75])
    assert all(abs(budget - expected) < 1e-15)


def test_mismanaged_mixed_calculation_of_budgets():
    dT_target = np.array([2.5, 3.5])
    zec = 0.4
    historical_dT = np.array([1, 0.95])
    non_co2_dT = np.array([0.1, 0.2, 0.3])
    tcre = 1
    earth_feedback_co2 = np.array([0.1, 0.2])
    error_message = re.escape(
        "operands could not be broadcast together with shapes (2,) (3,)"
    )
    with pytest.raises(ValueError, match=error_message):
        budget = src.budget_calculator_functions.calculate_budget(
            dT_target, zec, historical_dT, non_co2_dT, tcre, earth_feedback_co2
        )

def test_calculate_earth_syst():
    not_zero = 1000
    feedback = src.budget_calculator_functions.calculate_earth_system_feedback_co2(0, not_zero)
    assert feedback == 0
    two = 2
    feedback = src.budget_calculator_functions.calculate_earth_system_feedback_co2(two, not_zero)
    assert feedback == two * not_zero