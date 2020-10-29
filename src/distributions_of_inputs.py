import os
import netCDF4
import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate


def tcre_distribution(low, high, likelihood, n_return, tcre_dist):
    """

    :param low: float
        The lower limit of the probability distribution
    :param high: float
        The upper limit of the probability distribution. Must be larger than low. 
    :param likelihood: float
        Probability (between 0 and 1) that the value will be between low and high
    :param n_return: int
        The number of values to return
    :param tcre_dist: Str
        Either "normal" for a normal distribution, "lognormal" for a lognormal
        distribution, or "lognormal mean match", for a lognormal but with the same mean
        and standard deviation as a normal distribution fitting the same
        high/low/likelihood values.
    :return:
    """
    assert high > low, "High and low limits are the wrong way around"
    if tcre_dist == "normal":
        # We want a normal distribution such that we are between low and high with a
        # given likelihood.
        mean = (high + low) / 2
        # find the normalised z score that gives a normal cdf with given likelihood of
        # being between the required high and low values
        z = scipy.stats.norm.ppf((1 + likelihood) / 2)
        sd = (high - mean) / z
        return np.random.normal(mean, sd, n_return)
    elif tcre_dist == "lognormal mean match":
        mean = (high + low) / 2
        assert mean > 0, "lognormal distributions are always positive"
        z = scipy.stats.norm.ppf((1 + likelihood) / 2)
        sd = (high - mean) / z
        # The lognormal function takes arguments of the underlying mu and sigma values,
        # which are not the same as the actual mean and s.d., so we convert below
        # Derive relations from: mean = exp(\mu + \sigma^2/2),
        # sd = (exp(\sigma^2) - 1)^0.5 * exp(\mu + \sigma^2/2)
        sigma = (np.log(1 + (sd / mean) ** 2)) ** 0.5
        mu = np.log(mean) - sigma ** 2 / 2
        return np.random.lognormal(mean=mu, sigma=sigma, size=n_return)
    elif tcre_dist == "lognormal":
        assert high > 0
        assert low > 0
        # We have h = exp(\mu + \sigma z), l = exp(\mu - \sigma z) ,
        # for z normally distributed as before. Rearranging and solving:
        z = scipy.stats.norm.ppf((1 + likelihood) / 2)
        mu = 0.5 * np.log(low * high)
        sigma = 0.5 * np.log(high / low)
        return np.random.lognormal(mean=mu, sigma=sigma, size=n_return)
    # If you haven't returned yet, something went wrong.
    raise ValueError(
        "tcre_dist must be either normal, lognormal mean match or lognormal, it was {}".format(
            tcre_dist
        )
    )


def establish_median_temp_dep(models, temps):
    """
    Calculates the median line of fit
    :param models: pd.Database.
        Must include a column marked 'quantile' with a row 0.5
    :param temps: np.array.

    :return: pd.Series
    """
    return pd.Series(
        index=temps,
        data=models["b"].loc[models["quantile"] == 0.5].iloc[0] * temps
        + models["a"].loc[models["quantile"] == 0.5].iloc[0],
    )


def establish_least_sq_temp_dependence(db, temps, non_co2_col, temp_col):
    regres = np.polyfit(db[temp_col], db[non_co2_col], 1)
    return pd.Series(index=temps, data=temps * regres[0] + regres[1])


def load_data_from_MAGICC(
    non_co2_magicc_file, tot_magicc_file, yearfile, non_co2_col, tot_col,
        magicc_nonco2_temp_variable, tot_temp_variable, offset_years, peak_version=None
):
    """
    Loads the non-CO2 warming and total warming from files in the format output by
    MAGICC.
    :param non_co2_magicc_file: The file location for non-CO2 warming
    :param tot_magicc_file: The file for total warming
    :param yearfile: The file where emissions are recorded
    :param non_co2_col: The name of the non-CO2 warming column in the output
    :param tot_col: The name of the total warming column in the output df
    :param magicc_nonco2_temp_variable: The name of the non-CO2 warming variable to be
        investigated
    :param tot_temp_variable: The name of the total warming variable to be investigated
    :param offset_years: The years over which the baseline average should be taken
    :param peak_version: String describing which year the non-CO2 warming should be
        taken in.
    :return: pd.Dataframe
    """
    yeardf = pd.read_csv(yearfile, index_col=0)
    # Drop empty columns from the dataframe and calculate the year the emissions go to
    # zero
    empty_cols = [col for col in yeardf.columns if yeardf[col].isnull().all()]
    yeardf.drop(empty_cols,
            axis=1,
            inplace=True)
    scenario_cols = ["model", "region", "scenario"]
    del yeardf["unit"]
    total_co2 = yeardf.groupby(scenario_cols).sum()
    if not peak_version:
        total_co2.columns = [int(col[:4]) for col in total_co2.columns]
        zero_years = pd.Series(index=total_co2.index)
        for index, row in total_co2.iterrows():
            try:
                change_sign = np.where(row < 0)[0][0]
                zero_year = scipy.interpolate.interp1d(
                    [row.iloc[change_sign - 1], row.iloc[change_sign]],
                    [row.index[change_sign - 1], row.index[change_sign]]
                )(0)
                zero_years[index] = np.round(zero_year)
            except IndexError:
                del zero_years[index]
        # load temperature data and get it into the same format as the emissions data.
        # Do this for both non-CO2 and CO2-only temperature.
        temp_df = pd.DataFrame(
            index=zero_years.index, columns=[tot_col, non_co2_col], dtype=np.float64
        )
    else:
        temp_df = pd.DataFrame(
            index=total_co2.index, columns=[tot_col, non_co2_col], dtype=np.float64
        )
    for (warmingfile, col_name, temp_variable) in [
        (non_co2_magicc_file, non_co2_col, magicc_nonco2_temp_variable),
        (tot_magicc_file, tot_col, tot_temp_variable)
    ]:
        df = pd.read_csv(warmingfile, index_col=0)
        df = df.loc[df["variable"] == temp_variable]
        df.set_index(scenario_cols, drop=True, inplace=True)
        del df["unit"]
        del df["variable"]
        assert all([ind in df.index for ind in temp_df.index]), \
            "There is a mismatch between the emissions year file and the temperture " \
            "file"
        df = df.loc[[ind for ind in df.index if ind in temp_df.index]]
        df.columns = [int(col[:4]) for col in df.columns]
        # For each scenario, we subtract the average temperature from the offset years
        for ind, row in df.iterrows():
            if not peak_version:
                if warmingfile == non_co2_magicc_file:
                    temp = df.loc[ind][zero_years.loc[ind]]
                else:
                    temp = max(df.loc[ind])
            elif peak_version == "peakNonCO2Warming":
                temp = max(df.loc[ind])
            else:
                raise ValueError(
                    "Invalid choice for peak_version {}".format(peak_version)
                )
            temp_df[col_name][ind] = temp - df.loc[ind][offset_years].mean()

    return temp_df


def load_data_from_FaIR(
    folder_all,
    folder_co2_only,
    desired_scenarios_db,
    model_col,
    scenario_col,
    year_col,
    magicc_non_co2_col,
    magicc_temp_col,
    offset_years,
):
    all_files = os.listdir(folder_all)
    CO2_only_files = os.listdir(folder_co2_only)
    assert all_files == CO2_only_files
    # We must find the correspondence between the file systems and the known years of
    # peak emissions
    expected_filenames = (
        "IPCCSR15_"
        + desired_scenarios_db[model_col]
        + "_"
        + desired_scenarios_db[scenario_col]
    )
    assert len(expected_filenames) == len(
        set(expected_filenames)
    ), "Expected file names are not clearly distinguishable"
    expected_years = desired_scenarios_db[year_col]
    compare_filenames = [x.replace(" ", "_").replace(".", "_")[:-12] for x in all_files]
    assert len(compare_filenames) == len(
        set(compare_filenames)
    ), "Processed file names are not clearly distinguishable"
    files_found_indx = [(y in compare_filenames) for y in expected_filenames]
    expected_filenames = expected_filenames[files_found_indx]
    expected_years = expected_years[files_found_indx]
    temp_all_dbs = []
    temp_only_co2_dbs = []
    for ind in range(len(expected_filenames)):
        file = [
            i
            for (i, v) in zip(
                all_files,
                [expected_filenames.iloc[ind] == y for y in compare_filenames],
            )
            if v
        ]
        assert len(file) == 1
        file = file[0]
        open_link_all = netCDF4.Dataset(folder_all + file)
        time_ind = np.where(open_link_all.variables["time"] == expected_years.iloc[ind])
        assert len(time_ind) == 1, (
            "There must be exactly one match between the years"
            "and the expected year, but there are {}".format(sum(time_ind))
        )
        time_ind = time_ind[0]
        offset_inds = np.where(
            [y in offset_years for y in open_link_all.variables["time"][:]]
        )[0]
        assert len(offset_inds) == len(
            offset_years
        ), "We found the wrong number of offset years in the database."
        all_temp = (
            pd.DataFrame(open_link_all["temp"][time_ind, ::1]).mean(axis=0).mean(axis=0)
        )
        all_offset = (
            pd.DataFrame(open_link_all["temp"][offset_inds, ::1])
            .mean(axis=0)
            .mean(axis=0)
        )
        temp_all_dbs.append(all_temp - all_offset)
        open_link_co2_only = netCDF4.Dataset(folder_co2_only + file)
        only_co2_temp = (
            pd.DataFrame(open_link_co2_only["temp"][time_ind, ::1])
            .mean(axis=0)
            .mean(axis=0)
        )
        only_co2_offset = (
            pd.DataFrame(open_link_co2_only["temp"][offset_inds, ::1])
            .mean(axis=0)
            .mean(axis=0)
        )
        temp_only_co2_dbs.append(only_co2_temp - only_co2_offset)
        assert (
            all_offset > 0
            and all_temp > 0
            and only_co2_temp > 0
            and only_co2_offset > 0
        ), "Does the database really contain a negative temperature change?"
    temp_no_co2_dbs = np.array(temp_all_dbs) - temp_only_co2_dbs
    assert all(x > 0 for x in temp_all_dbs) and all(
        x > 0 for x in temp_only_co2_dbs
    ), "Does the database really contain a negative temperature change?"
    dbs = pd.DataFrame(
        {magicc_temp_col: temp_all_dbs, magicc_non_co2_col: temp_no_co2_dbs},
        dtype="float32",
    )
    return dbs
