import numpy as np
import pandas as pd

def tcre_distribution(mean, sd, n_return=1):
    return np.random.normal(mean, sd, n_return)


def establish_temp_dependence(file, temps):
    non_co2_col = "non-co2 warming (rel. to 2010-2019) at peak cumulative emissions co2 (rel. to 2015-2015)"
        #"non-co2 warming (rel. to 2010-2019) at peak surface temperature (rel. to 2010-2019)"
    temp_col = "peak surface temperature (rel. to 2010-2019)"
    db = pd.read_csv(file)
    regres = np.polyfit(db[temp_col], db[non_co2_col], 1)
    return pd.Series(index=temps, data=temps * regres[0] + regres[1])
