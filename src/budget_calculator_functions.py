def calculate_budget(
        dT_target,
        zec,
        historical_dT,
        non_co2_dT,
        tcre,
        earth_feedback_co2,
):
    """

    :param dT_target: The target temperature change to achieve
    :param zec: The change in temperature that will occur after zero emissions has been
    reached.
    :param historical_dT: The temperature difference already seen
    :param non_co2_dT: The non-carbon contributions to temperature change
    :param tcre: Transient climate response to cumulative carbon emissions
    :param earth_feedback_co2: CO2 emissions from temperature-dependent Earth feedback
    loops
    :return: the remaining CO2 budget.
    """

    remaining_dT = dT_target - zec - non_co2_dT - historical_dT
    return remaining_dT/tcre - earth_feedback_co2

