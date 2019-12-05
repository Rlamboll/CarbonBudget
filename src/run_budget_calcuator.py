import numpy as np
import pandas as pd
import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as budget_func
import matplotlib.pyplot as plt

# Input values
# ______________________________________________________________________________________

# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.2, 0.1)
# The number of loops performed for each temperature
n_loops = 5000000
# The change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec = 0.0
# The temperature difference already seen. (Units: C)
historical_dT = 1.0
# The distribution of the TCRE function - either "normal". "lognormal mean match" or
# "lognormal likely". The latter two cases are lognormal distributions, in the first
# case matching the mean and sd of the normal distribution, in the second case matching
# the likelihood
tcre_dist = "lognormal mean match"
# The mean of the distribution of TCRE. We use units of C per GtCO2.
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 0.2 / 1000
# The standard deviation of the distribution of TCRE.
tcre_high = 0.7 / 1000
# likelihood is the probability that results fit between the low and high value
likelihood = 0.6827
# CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# (Units: GtCO2/C)
earth_feedback_co2_per_C = 0
# Any emissions that have taken place too recently to have factored into the measured
# temperature change, and therefore must be subtracted from the budget (Units: GtCO2)
recent_emissions = 290
# We will present the budgets at these quantiles of the TCRE.
quantiles_to_report = np.array([0.33, 0.5, 0.66])
# Output file location for budget data.
output_file = "../Output/budget_calculation.csv"
# Output location for figure of peak warming vs non-CO2 warming
output_figure_file = "../Output/non_co2_cont_to_peak_warming.png"

#   Information for reading in files used to calculate non-CO2 component:
#       MAGICC files

# The file in which we find the MAGICC model estimate for the non-carbon contributions
# to temperature change. (Units: C)
non_co2_magicc_file = "../InputData/AR6comptibleMAGICCsetup.csv"
# The names of the columns of interest in the MAGICC model file
magicc_non_co2_col = "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2 (rel. to 2015-2015)"
# The name of the column containing the surface temperature of interest
magicc_temp_col = "peak surface temperature (rel. to 2010-2019)"
# Names of the model, scenario and year columns in the MAGICC database
model_col = "model"
scenario_col = "scenario"
year_col = "peak cumulative emissions co2 (rel. to 2015-2015) year"

#       FaIR files
# The folders for the unscaled anthropological temperature changes files (many nc files)
fair_anthro_folder = "../InputData/fair141_sr15_ar6fodsetup/FAIR141anthro_unscaled/"
fair_co2_only_folder = "../InputData/fair141_sr15_ar6fodsetup/FAIR141CO2_unscaled/"
# Years over which we set the average temperature to 0. This should be the same as the
# MAGICC data uses. Note that the upper limit of the range is not included in python.
fair_offset_years = np.arange(2010, 2020, 1)

# ______________________________________________________________________________________
# The parts below should not need editing

budget_quantiles = pd.DataFrame(index=dT_targets, columns=quantiles_to_report)
budget_quantiles.index.name = "dT_targets"
# We interpret the higher quantiles as meaning a smaller budget
quantiles_to_report = 1 - quantiles_to_report
magicc_db = distributions.load_data_from_MAGICC(
    non_co2_magicc_file,
    magicc_non_co2_col,
    magicc_temp_col,
    model_col,
    scenario_col,
    year_col,
)
non_co2_dT_fair = distributions.load_data_from_FaIR(
    fair_anthro_folder,
    fair_co2_only_folder,
    magicc_db,
    model_col,
    scenario_col,
    year_col,
    magicc_non_co2_col,
    magicc_temp_col,
    fair_offset_years,
)
all_non_co2_db = magicc_db[[magicc_non_co2_col, magicc_temp_col]].append(
    non_co2_dT_fair
)
non_co2_dTs = distributions.establish_temp_dependence(
    all_non_co2_db, dT_targets - historical_dT, magicc_non_co2_col, magicc_temp_col
)
# TODO: include FaIR in non-CO2 estimates
for dT_target in dT_targets:
    earth_feedback_co2 = budget_func.calculate_earth_system_feedback_co2(
        dT_target, earth_feedback_co2_per_C
    )
    non_co2_dT = non_co2_dTs.loc[dT_target - historical_dT]
    tcres = distributions.tcre_distribution(tcre_low, tcre_high, likelihood, n_loops, tcre_dist)
    budgets = budget_func.calculate_budget(
        dT_target, zec, historical_dT, non_co2_dT, tcres, earth_feedback_co2
    )
    budgets = budgets - recent_emissions
    budget_quantiles.loc[dT_target] = np.quantile(budgets, quantiles_to_report)

# Save output
budget_quantiles.to_csv(output_file)

plt.close()
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
plt.scatter(magicc_db[magicc_temp_col], magicc_db[magicc_non_co2_col])
plt.scatter(non_co2_dT_fair[magicc_temp_col], non_co2_dT_fair[magicc_non_co2_col])
plt.legend(["MAGICC", "FaIR"])
plt.ylabel(magicc_non_co2_col)
plt.xlabel(magicc_temp_col)
plt.savefig(output_figure_file)
