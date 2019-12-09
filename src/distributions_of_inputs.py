import os

import netCDF4
import numpy as np
import pandas as pd
import scipy.stats


def tcre_distribution(low, high, likelihood, n_return, tcre_dist):
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


def establish_temp_dependence(db, temps, non_co2_col, temp_col):
    regres = np.polyfit(db[temp_col], db[non_co2_col], 1)
    return pd.Series(index=temps, data=temps * regres[0] + regres[1])


def load_data_from_MAGICC(
    file, non_co2_col, temp_col, model_col, scenario_col, year_col
):
    df = pd.read_csv(file)
    df = df.loc[df["quantile"] == 0.5]
    df = df[[non_co2_col, temp_col, model_col, scenario_col, year_col]]
    assert len(df) == len(df.groupby([model_col, scenario_col]))
    return df


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
