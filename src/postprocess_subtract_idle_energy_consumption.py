

def subtract_baseline_energy_consumption(df, baseline_energy_consumption=3):
    """
    Subtract the baseline energy consumption from the energy consumption of the device.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the energy consumption of the device.
    baseline_energy_consumption : float
        Baseline energy consumption of the device in Watts.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the energy consumption of the device with the baseline energy consumption subtracted.
    """

    df['energy_consumption_joules_raw'] = df['energy_consumption_joules']
    df['energy_consumption_joules'] = df['energy_consumption_raw'] - baseline_energy_consumption
    return df