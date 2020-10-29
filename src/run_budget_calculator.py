import numpy as np
import pandas as pd
import src.distributions_of_inputs as distributions
import src.budget_calculator_functions as budget_func
import matplotlib.pyplot as plt

# Input values
# ______________________________________________________________________________________

# The target temperature changes to achieve. (Units: C)
dT_targets = np.arange(1.1, 2.6, 0.1)
# The number of loops performed for each temperature
n_loops = 10000000
# The change in temperature that will occur after zero emissions has been reached.
# (Units: C)
zec = 0.0
# The temperature difference already seen. (Units: C)
historical_dT = 1.1
# The distribution of the TCRE function - either "normal". "lognormal mean match" or
# "lognormal". The latter two cases are lognormal distributions, in the first
# case matching the mean and sd of the normal distribution which fits the likelihood,
# in the second case matching the likelihood.
tcre_dist = "normal"
# The upper and lower bounds of the distribution of TCRE. We use units of C per GtCO2.
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 1.0 / 3664
tcre_high = 2.1 / 3664
# likelihood is the probability that results fit between the low and high value
likelihood = 0.6827
# CO2 emissions per degree C from temperature-dependent Earth feedback loops.
# (Units: GtCO2/C)
earth_feedback_co2_per_C = 135
# Any emissions that have taken place too recently to have factored into the measured
# temperature change, and therefore must be subtracted from the budget (Units: GtCO2)
recent_emissions = 0
# We will present the budgets at these quantiles of the TCRE.
quantiles_to_report = np.array([0.17, 0.33, 0.5, 0.66, 0.83])
# Name of the output folder
output_folder = "../Output/ar6draft2/"
# Output file location for budget data. Includes {} sections detailing inclusion of
# TCRE, inclusion of magic/fair, earth system feedback and likelihood. More added later
output_file = output_folder + \
              "budget_calculation_{}_magicc_{}_fair_{}_earthsfb_{}_likelihood_{}"
# Output location for figure of peak warming vs non-CO2 warming. More appended later
output_figure_file = output_folder + "non_co2_cont_to_peak_warming_magicc_{}_fair_{}"
# Quantile fit lines to plot on the graph.
# If use_median_non_co2 == True, this must include 0.5, as we use this value
quantiles_to_plot = [0.05, 0.5, 0.95]
# How should we dot these lines? This list must be as long as quantiles_to_plot.
line_dotting = ["--", "-", "--"]
# Should we use the median regression or the least-squares best fit for the non-CO2
# relationship?
use_median_non_co2 = True
# Where should we save the results of the figure with trend lines? Not plotted if
# use_median_non_co2 is True.
output_all_trends = output_folder + "TrendLinesWithMagicc.pdf"

#       Information for reading in files used to calculate non-CO2 component:

#       MAGICC files
# Should we use a variant means of measuring the non-CO2 warming?
# Default = None; ignore scenarios with non-peaking cumulative CO2 emissions, use
# non-CO2 warming in the year of peak cumulative CO2.
# If "peakNonCO2Warming", we use the highest non-CO2 temperature,
# irrespective of emissions peak.
# If "nonCO2AtPeakTot", computes the non-CO2 component at the time of peak total
# temperature.
peak_version = None  # "nonCO2AtPeakTot"
output_file += "_" + str(peak_version) + ".csv"
output_figure_file += "_" + str(peak_version) + ".pdf"
# The folder and files in which we find the MAGICC model estimate for the non-carbon and
# carbon contributions to temperature change.
input_folder = "../InputData/Non-CO2 - AR6 emulator SR15 scenarios/"
non_co2_magicc_file = input_folder + "nonco2_results_20201026-sr15-nonco2_GSAT-Non-CO2.csv"
tot_magicc_file = input_folder + "nonco2_results_20201026-sr15-nonco2_GSAT.csv"
# The file in which we find the emissions data
emissions_file = input_folder + "nonco2_results_20201026-sr15-nonco2_Emissions-CO2.csv"
# The name of the non-CO2 warming column output from in the MAGICC model file analysis
magicc_non_co2_col = "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2"
# The name of the peak temperature column output
magicc_temp_col = "peak surface temperature (rel. to 2010-2019)"
# The names of the temperature variables in MAGICC files (also specifies the quantile)
magicc_nonco2_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.4.1|50.0th Percentile"
magicc_tot_temp_variable = "SR15 climate diagnostics|Raw Surface Temperature (GSAT)|MAGICCv7.4.1|50.0th Percentile"
# Do we want to save the output of the MAGICC analysis? If so, give a file name with a
# variable in it. Otherwise leave as None
magicc_savename = output_folder + "magicc_nonCO2_temp" + str(peak_version) + ".csv"
# Years over which we set the average temperature to 0.
# Note that the upper limit of the range is not included in python.
temp_offset_years = np.arange(2010, 2020, 1)

# ______________________________________________________________________________________
# The parts below should not need editing

magicc_db = distributions.load_data_from_MAGICC(
    non_co2_magicc_file,
    tot_magicc_file,
    emissions_file,
    magicc_non_co2_col,
    magicc_temp_col,
    magicc_nonco2_temp_variable,
    magicc_tot_temp_variable,
    temp_offset_years,
    peak_version,
)
if magicc_savename:
    magicc_db.to_csv(magicc_savename)
non_co2_dT_fair = np.nan  # We currently do not consider the impact of the FaIR model
# We interpret the higher quantiles as meaning a smaller budget
inverse_quantiles_to_report = 1 - quantiles_to_report
# Construct the container for saved results
all_fit_lines = []
# Modify the following loop to use subsets of data for robustness checks
for case_ind in range(1):
    include_magicc = True
    include_fair = False

    budget_quantiles = pd.DataFrame(index=dT_targets, columns=quantiles_to_report)
    budget_quantiles.index.name = "dT_targets"

    if include_fair and include_magicc:
        all_non_co2_db = magicc_db[[magicc_non_co2_col, magicc_temp_col]].append(
            non_co2_dT_fair
        )
    elif include_fair:
        all_non_co2_db = non_co2_dT_fair
    elif include_magicc:
        all_non_co2_db = magicc_db[[magicc_non_co2_col, magicc_temp_col]]
    else:
        raise ValueError("You must include either magicc or fair data")
    if use_median_non_co2:
        assert 0.5 in quantiles_to_plot, (
            "The median value, quantiles_to_plot=0.5, "
            "must be included if use_median_non_co2==True"
        )
        # The quantile regression program is temperamental, so we ensure the data has
        # the correct numeric format before passing it
        x = all_non_co2_db[magicc_temp_col].astype(np.float64)
        y = all_non_co2_db[magicc_non_co2_col].astype(np.float64)
        xy_df = pd.DataFrame({"x": x, "y": y})
        xy_df = xy_df.reset_index(drop=True)
        quantile_reg_trends = budget_func.quantile_regression_find_relationships(
            xy_df, quantiles_to_plot
        )
        non_co2_dTs = distributions.establish_median_temp_dep(
            quantile_reg_trends, dT_targets - historical_dT
        )
    else:
        # If not quantile regression, we use the least squares fit to the non-CO2 data
        non_co2_dTs = distributions.establish_least_sq_temp_dependence(
            all_non_co2_db,
            dT_targets - historical_dT,
            magicc_non_co2_col,
            magicc_temp_col,
        )

    for dT_target in dT_targets:
        earth_feedback_co2 = budget_func.calculate_earth_system_feedback_co2(
            dT_target - historical_dT, earth_feedback_co2_per_C
        )
        non_co2_dT = non_co2_dTs.loc[dT_target - historical_dT]
        tcres = distributions.tcre_distribution(
            tcre_low, tcre_high, likelihood, n_loops, tcre_dist
        )
        budgets = budget_func.calculate_budget(
            dT_target, zec, historical_dT, non_co2_dT, tcres, earth_feedback_co2
        )
        budgets = budgets - recent_emissions
        budget_quantiles.loc[dT_target] = np.quantile(
            budgets, inverse_quantiles_to_report
        )

    # Save output in the correct format
    budget_quantiles = budget_quantiles.reset_index()
    budget_quantiles["Future_warming"] = budget_quantiles["dT_targets"] - historical_dT
    budget_quantiles = budget_quantiles.set_index("Future_warming")
    budget_quantiles.to_csv(
        output_file.format(
            tcre_dist,
            include_magicc,
            include_fair,
            earth_feedback_co2_per_C,
            likelihood,
        )
    )

    # Make plots of the data
    temp_plot_limits = [
        min(magicc_db[magicc_temp_col]),
        max(magicc_db[magicc_temp_col]),
    ]
    non_co2_plot_limits = [
        min(magicc_db[magicc_non_co2_col]),
        max(magicc_db[magicc_non_co2_col]),
    ]

    def add_fringe(limits, fringe):
        # Helper function for adding a small amount either side of the limits
        assert len(limits) == 2
        offset = fringe * (limits[1] - limits[0])
        limits[0] = limits[0] - offset
        limits[1] = limits[1] + offset
        return limits
    # 0.04 is chosen for the fringes for aesthetic reasons
    temp_plot_limits = add_fringe(temp_plot_limits, 0.04)
    non_co2_plot_limits = add_fringe(non_co2_plot_limits, 0.04)
    plt.close()
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    legend_text = []
    if include_magicc:
        plt.scatter(
            magicc_db[magicc_temp_col], magicc_db[magicc_non_co2_col], color="blue"
        )
        legend_text.append("MAGICC")
    if include_fair:
        plt.scatter(
            non_co2_dT_fair[magicc_temp_col],
            non_co2_dT_fair[magicc_non_co2_col],
            color="orange",
        )
        legend_text.append("FaIR")
    plt.xlim(temp_plot_limits)
    plt.ylim(non_co2_plot_limits)
    plt.legend(legend_text)
    plt.ylabel(magicc_non_co2_col)
    plt.xlabel(magicc_temp_col)
    if not use_median_non_co2:
        x = all_non_co2_db[magicc_temp_col]
        y = all_non_co2_db[magicc_non_co2_col]
        equation_of_fit = np.polyfit(x, y, 1)
        all_fit_lines.append(equation_of_fit)
        plt.plot(np.unique(x), np.poly1d(equation_of_fit)(np.unique(x)), color="black")
        equation_text = "y = " + str(round(equation_of_fit[0], 4)) + "x" " + " + str(
            round(equation_of_fit[1], 4)
        )
        plt.text(
            0.9,
            0.1,
            equation_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        quantiles_of_plot = budget_func.rolling_window_find_quantiles(
            xs=all_non_co2_db[magicc_temp_col],
            ys=all_non_co2_db[magicc_non_co2_col],
            quantiles=quantiles_to_plot,
            nwindows=10,
        )
        x = quantiles_of_plot.index.values
        for col in quantiles_of_plot.columns:
            y = quantiles_of_plot[col].values
            if col == 0.5:
                dashes = [1, 0]
                color = "black"
            else:
                dashes = [6, 2]
                color = "grey"
            plt.plot(
                np.unique(x),
                np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
                dashes=dashes,
                color=color,
            )
    else:
        minT = temp_plot_limits[0]
        maxT = temp_plot_limits[1]
        for i in range(len(quantile_reg_trends)):
            plt.plot(
                (minT, maxT),
                (
                    quantile_reg_trends["b"][i] * minT + quantile_reg_trends["a"][i],
                    quantile_reg_trends["b"][i] * maxT + quantile_reg_trends["a"][i],
                ),
                ls=line_dotting[i],
                color="black",
            )
    fig.savefig(
        output_figure_file.format(include_magicc, include_fair), bbox_inches="tight"
    )
plt.close()
# Plot all trendlines together, if data was processed without median values
if not use_median_non_co2:
    x = temp_plot_limits
    y = non_co2_plot_limits
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    colours = ["black", "blue", "orange"]
    for eqn in all_fit_lines:
        plt.plot(np.unique(x), np.poly1d(eqn)(np.unique(x)), color=colours[0])
        colours = colours[1:]
    plt.xlim(temp_plot_limits)
    plt.ylim(non_co2_plot_limits)
    plt.legend(["MAGICC and FaIR", "MAGICC only", "FaIR only"])
    plt.ylabel(magicc_non_co2_col)
    plt.xlabel(magicc_temp_col)
    fig.savefig(output_all_trends)

print("The analysis has completed.")