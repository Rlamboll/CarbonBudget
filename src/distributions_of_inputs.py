import os

import netCDF4
import numpy as np
import pandas as pd


def tcre_distribution(mean, sd, n_return=1):
    return np.random.normal(mean, sd, n_return)


def establish_temp_dependence(db, temps, non_co2_col, temp_col):
    regres = np.polyfit(db[temp_col], db[non_co2_col], 1)
    return pd.Series(index=temps, data=temps * regres[0] + regres[1])


def load_data_from_MAGICC(
        file,
        non_co2_col,
        temp_col,
        model_col,
        scenario_col,
        year_col,
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
):
    all_files = os.listdir(folder_all)
    CO2_only_files = os.listdir(folder_co2_only)
    assert all_files == CO2_only_files
    # We must find the correspondence between the file systems and the known years of
    # peak emissions
    expected_filenames = "IPCCSR15_" + desired_scenarios_db[model_col] + "_" + \
                         desired_scenarios_db[scenario_col]
    assert len(expected_filenames) == len(set(expected_filenames)), \
        "Expected file names are not clearly distinguishable"
    expected_years = desired_scenarios_db[year_col]
    compare_filenames = [x.replace(" ", "_").replace(".", "_")[:-12] for x in all_files]
    assert len(compare_filenames) == len(set(compare_filenames)), \
        "Processed file names are not clearly distinguishable"
    files_found_indx = [(y in compare_filenames) for y in expected_filenames]
    expected_filenames = expected_filenames[files_found_indx]
    expected_years = expected_years[files_found_indx]
    temp_all_dbs = []
    temp_only_co2_dbs = []
    for ind in range(len(expected_filenames)):
        file = [i for (i, v) in zip(all_files, [
            expected_filenames.iloc[ind] == y for y in compare_filenames
        ]) if v]
        assert len(file) == 1
        file = file[0]
        open_link_all = netCDF4.Dataset(folder_all + file)
        time_ind = np.where(open_link_all.variables["time"] == expected_years.iloc[ind])
        assert len(time_ind) == 1, "There must be exactly one match between the years" \
                                   "and the expected year, but there are {}" \
            .format(sum(time_ind))
        time_ind = time_ind[0]
        temp_all_dbs.append(
            pd.DataFrame(open_link_all["temp"][time_ind, ::1]).mean(axis=1)
        )
        open_link_co2_only = netCDF4.Dataset(folder_co2_only + file)
        temp_only_co2_dbs.append(
            pd.DataFrame(open_link_co2_only["temp"][time_ind, ::1]).mean(axis=1))
        # TODO: calculate offsets
    dbs = pd.DataFrame({"temp_all": temp_all_dbs, "temp_CO2": temp_only_co2_dbs}, dtype="float32")
    return dbs
