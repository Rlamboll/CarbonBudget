import numpy as np
import pandas as pd
import src.distributions_of_inputs
import src.budget_calculator_functions


# Input values
# ______________________________________________________________________________________
# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.2, 0.1)
# The number of loops performed for each temperature
n_loops = 1000000
# The change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec = 0.0
# The temperature difference already seen. (Units: C)
historical_dT = 1.0
# The non-carbon contributions to temperature change. (Units: C)
non_co2_dT = 0.15
# The mean of the distribution of TCRE. We use units of C per GtCO2.
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_mean = (0.2 + 0.7) / 2000
# The standard deviation of the distribution of TCRE.
tcre_sd = (0.7 - 0.2) / 2000
# CO2 emissions from temperature-dependent Earth feedback loops. (Units: GtCO2)
earth_feedback_co2 = 100
# We will present the budgets at these quantiles.
quantiles_to_report = np.array([0.33, 0.5, 0.66])
# Output file location.
output_file = "../Output/budget_calculation.csv"

# ______________________________________________________________________________________

# The parts below should not need editing

budget_quantiles = pd.DataFrame(index=dT_targets, columns=quantiles_to_report)
budget_quantiles.index.name = "dT_targets"
# We interpret the higher quantiles as meaning a smaller budget
quantiles_to_report = 1 - quantiles_to_report

for dT_target in dT_targets:
    tcres = src.distributions_of_inputs.tcre_distribution(tcre_mean, tcre_sd, n_loops)

    budgets = src.budget_calculator_functions.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcres, earth_feedback_co2
    )
    budget_quantiles.loc[dT_target] = np.quantile(budgets, quantiles_to_report)

# Save output
budget_quantiles.to_csv(output_file)
