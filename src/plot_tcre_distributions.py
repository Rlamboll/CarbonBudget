from src.distributions_of_inputs import tcre_distribution
import matplotlib.pyplot as plt
import numpy as np

tcre_dists = ["normal", "lognormal"]
# The mean of the distribution of TCRE. We use units of C per GtCO2.
# (TCRE = Transient climate response to cumulative carbon emissions)
tcre_low = 1.1 / 3664
# The standard deviation of the distribution of TCRE.
tcre_high = 2.2 / 3664
# likelihood is the probability that results fit between the low and high value
likelihood = 0.6827
# Number to return such that the granularity is negligible.
n_runs = 500000000
# Save file location
save_location = "../Output/tcre_distribution_{}_low_{}_high_{}_likelihood_{}.pdf"

for tcre_dist in tcre_dists:
    xs = tcre_distribution(tcre_low, tcre_high, likelihood, n_runs, tcre_dist)
    normed = np.histogram(xs, bins=800, density=True)
    plt.close()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(normed[1][1:], normed[0])
    plt.xlabel(u'\N{DEGREE SIGN}' + "C/Gt CO2")
    plt.ylabel("Probability density")
    plt.xlim([-0.00025, 0.00125])
    plt.ylim([0, 3000])
    fig.savefig(save_location.format(tcre_dist, tcre_low, tcre_high, likelihood))

