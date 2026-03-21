import os
import numpy as np
import pandas as pd

def aggregating_nominal_data(values, n):
    return np.array(values).reshape(-1, n).sum(axis=1).tolist()

def aggregating_std_data(values, n):
    return np.sqrt((np.array(values) ** 2).reshape(-1, n).sum(axis=1)).tolist()

def aggregating_nominal_cop(values, n):
    return np.array(values).reshape(-1, n).mean(axis=1).tolist()

def aggregating_std_cop(values, n):
    return (np.sqrt((np.array(values) ** 2).reshape(-1, n).sum(axis=1)) / n).tolist()

def fetch_raw_data():
    nominal_xls = pd.ExcelFile(os.path.join('Data', 'Operational Data Nominal.xlsx'))
    daily_std_xls = pd.ExcelFile(os.path.join('Data', 'Operational Data Daily Std.xlsx'))
    cluster_std_xls = pd.ExcelFile(os.path.join('Data', 'Operational Data Cluster Std.xlsx'))

    electricity_demand_2023 = nominal_xls.parse('2023 Hourly Electricity Demand')['Consumption (kWh/h)']
    electricity_demand_2023 = electricity_demand_2023.where(electricity_demand_2023 >= 100).interpolate(method='linear').bfill().ffill().tolist()
    electricity_demand_2024 = nominal_xls.parse('2024 Hourly Electricity Demand')['Consumption (kWh/h)']
    electricity_demand_2024 = electricity_demand_2024.where(electricity_demand_2024 >= 100).interpolate(method='linear').bfill().ffill().tolist()

    daily_std_2023 = daily_std_xls.parse('2023 Hourly Electricity Demand').fillna(0.0)
    cluster_std_2023 = cluster_std_xls.parse('2023 Hourly Electricity Demand').fillna(0.0)
    sigma_d_2023_elec = daily_std_2023.iloc[:, 0].values
    sigma_c_2023_elec = cluster_std_2023.iloc[:, 0].values
    combined_std_2023_elec = (sigma_d_2023_elec * sigma_c_2023_elec)[24:]

    daily_std_2024 = daily_std_xls.parse('2024 Hourly Electricity Demand').fillna(0.0)
    cluster_std_2024 = cluster_std_xls.parse('2024 Hourly Electricity Demand').fillna(0.0)
    sigma_d_2024_elec = daily_std_2024.iloc[:, 0].values
    sigma_c_2024_elec = cluster_std_2024.iloc[:, 0].values
    combined_std_2024_elec = (sigma_d_2024_elec * sigma_c_2024_elec)[:(8760-24)]

    base_electricity_demand = [(val_2023 + val_2024) / 2 for val_2023, val_2024 in zip(electricity_demand_2023[24:], electricity_demand_2024[:(8760-24)])]
    base_electricity_demand_std = [0.5 * np.sqrt(s1**2 + s2**2) for s1, s2 in zip(combined_std_2023_elec, combined_std_2024_elec)]

    heat_demand_2024 = nominal_xls.parse('2024 Hourly Heat Demand')['Consumption (kWh/h)']
    heat_demand_2024 = heat_demand_2024.where(heat_demand_2024 >= 100).interpolate(method='linear').bfill().ffill().tolist()
    base_heat_demand = heat_demand_2024[:(8760-24)]

    daily_std_2024_heat = daily_std_xls.parse('2024 Hourly Heat Demand').fillna(0.0)
    cluster_std_2024_heat = cluster_std_xls.parse('2024 Hourly Heat Demand').fillna(0.0)
    sigma_d_2024_heat = daily_std_2024_heat.iloc[:, 0].values
    sigma_c_2024_heat = cluster_std_2024_heat.iloc[:, 0].values
    base_heat_demand_std = (sigma_d_2024_heat * sigma_c_2024_heat)[:(8760-24)]

    solar_xls = pd.ExcelFile(os.path.join('Data', 'Solar Power.xlsx'))
    solar_initial = solar_xls.parse('Initial values')
    solar_advancements = {1: solar_xls.parse('Advancements1'), 2: solar_xls.parse('Advancements2'), 3: solar_xls.parse('Advancements3')}
    nominal_solar = nominal_xls.parse('solar').values[:(8760-24)]
    daily_std_solar = daily_std_xls.parse('solar').fillna(0.0)
    cluster_std_solar = cluster_std_xls.parse('solar').fillna(0.0)
    combined_std_solar = (daily_std_solar.values * cluster_std_solar.values)[:(8760-24)]
    base_solar_periodic_generation = nominal_solar.T.tolist()
    base_solar_periodic_generation_std = combined_std_solar.T.tolist()

    wind_xls = pd.ExcelFile(os.path.join('Data', 'Wind Power.xlsx'))
    wind_initial = wind_xls.parse('Initial values')
    wind_advancements = {1: wind_xls.parse('Advancements1')}
    nominal_wind = nominal_xls.parse('wind').values[:(8760-24)]
    daily_std_wind = daily_std_xls.parse('wind').fillna(0.0)
    cluster_std_wind = cluster_std_xls.parse('wind').fillna(0.0)
    combined_std_wind = (daily_std_wind.values * cluster_std_wind.values)[:(8760-24)]
    base_wind_periodic_generation = nominal_wind.T.tolist()
    base_wind_periodic_generation_std = combined_std_wind.T.tolist()

    el_storage_xls = pd.ExcelFile(os.path.join('Data', 'Electricity Storage.xlsx'))
    electricity_storage_initial = el_storage_xls.parse('Initial values')
    electricity_storage_advancements = {1: el_storage_xls.parse('Advancements1'), 2: el_storage_xls.parse('Advancements2'), 3: el_storage_xls.parse('Advancements3')}

    pt_xls = pd.ExcelFile(os.path.join('Data', 'Parabolic Trough.xlsx'))
    parabolic_trough_initial = pt_xls.parse('Initial values')
    parabolic_trough_advancements = {1: pt_xls.parse('Advancements1')}
    nominal_pt = nominal_xls.parse('parabolic trough').values[:(8760-24)]
    daily_std_pt = daily_std_xls.parse('parabolic trough').fillna(0.0)
    cluster_std_pt = cluster_std_xls.parse('parabolic trough').fillna(0.0)
    combined_std_pt = (daily_std_pt.values * cluster_std_pt.values)[:(8760-24)]
    base_parabolic_trough_periodic_generation = nominal_pt.T.tolist()
    base_parabolic_trough_periodic_generation_std = combined_std_pt.T.tolist()

    hp_xls = pd.ExcelFile(os.path.join('Data', 'Heat Pump.xlsx'))
    heat_pump_initial = hp_xls.parse('Initial values')
    heat_pump_advancements = {1: hp_xls.parse('Advancements1')}
    nominal_hp = nominal_xls.parse('heat pump').values[:(8760-24)]
    daily_std_hp = daily_std_xls.parse('heat pump').fillna(0.0)
    cluster_std_hp = cluster_std_xls.parse('heat pump').fillna(0.0)
    combined_std_hp = (daily_std_hp.values * cluster_std_hp.values)[:(8760-24)]
    base_heat_pump_cop_nominal = nominal_hp.T.tolist()
    base_heat_pump_cop_std = combined_std_hp.T.tolist()

    hs_xls = pd.ExcelFile(os.path.join('Data', 'Heat Storage.xlsx'))
    heat_storage_initial = hs_xls.parse('Initial values')
    heat_storage_advancements = {1: hs_xls.parse('Advancements1')}

    return dict(
        base_electricity_demand=base_electricity_demand, base_electricity_demand_std=base_electricity_demand_std,
        base_heat_demand=base_heat_demand, base_heat_demand_std=base_heat_demand_std.tolist(),
        solar_initial=solar_initial, solar_advancements=solar_advancements,
        base_solar_periodic_generation=base_solar_periodic_generation, base_solar_periodic_generation_std=base_solar_periodic_generation_std,
        wind_initial=wind_initial, wind_advancements=wind_advancements,
        base_wind_periodic_generation=base_wind_periodic_generation, base_wind_periodic_generation_std=base_wind_periodic_generation_std,
        electricity_storage_initial=electricity_storage_initial, electricity_storage_advancements=electricity_storage_advancements,
        parabolic_trough_initial=parabolic_trough_initial, parabolic_trough_advancements=parabolic_trough_advancements,
        base_parabolic_trough_periodic_generation=base_parabolic_trough_periodic_generation, base_parabolic_trough_periodic_generation_std=base_parabolic_trough_periodic_generation_std,
        heat_pump_initial=heat_pump_initial, heat_pump_advancements=heat_pump_advancements,
        base_heat_pump_cop_nominal=base_heat_pump_cop_nominal, base_heat_pump_cop_std=base_heat_pump_cop_std,
        heat_storage_initial=heat_storage_initial, heat_storage_advancements=heat_storage_advancements,
    )

def fetch_data(numStages, numSubperiods, numSubterms, epsilon=0, raw_data=None):
    if raw_data is None:
        raw_data = fetch_raw_data()

    discount_factor = 0.97
    subterm_length = int((8760-24)/numSubterms)

    electricity_demand_nominal = aggregating_nominal_data(raw_data['base_electricity_demand'], subterm_length)
    electricity_demand_std = aggregating_std_data(raw_data['base_electricity_demand_std'], subterm_length)
    electricity_demand = [nom + epsilon * std for nom, std in zip(electricity_demand_nominal, electricity_demand_std)]

    heat_demand_nominal = aggregating_nominal_data(raw_data['base_heat_demand'], subterm_length)
    heat_demand_std = aggregating_std_data(raw_data['base_heat_demand_std'], subterm_length)
    heat_demand = [nom + epsilon * std for nom, std in zip(heat_demand_nominal, heat_demand_std)]

    solar_periodic_generation_nominal = [aggregating_nominal_data(row, subterm_length) for row in raw_data['base_solar_periodic_generation']]
    solar_periodic_generation_std = [aggregating_std_data(row, subterm_length) for row in raw_data['base_solar_periodic_generation_std']]
    solar_periodic_generation = [[max(0, nom - epsilon * std) for nom, std in zip(n_row, s_row)] for n_row, s_row in zip(solar_periodic_generation_nominal, solar_periodic_generation_std)]

    wind_periodic_generation_nominal = [aggregating_nominal_data(row, subterm_length) for row in raw_data['base_wind_periodic_generation']]
    wind_periodic_generation_std = [aggregating_std_data(row, subterm_length) for row in raw_data['base_wind_periodic_generation_std']]
    wind_periodic_generation = [[max(0, nom - epsilon * std) for nom, std in zip(n_row, s_row)] for n_row, s_row in zip(wind_periodic_generation_nominal, wind_periodic_generation_std)]

    parabolic_trough_periodic_generation_nominal = [aggregating_nominal_data(row, subterm_length) for row in raw_data['base_parabolic_trough_periodic_generation']]
    parabolic_trough_periodic_generation_std = [aggregating_std_data(row, subterm_length) for row in raw_data['base_parabolic_trough_periodic_generation_std']]
    parabolic_trough_periodic_generation = [[max(0, nom - epsilon * std) for nom, std in zip(n_row, s_row)] for n_row, s_row in zip(parabolic_trough_periodic_generation_nominal, parabolic_trough_periodic_generation_std)]

    heat_pump_cop_nominal = [aggregating_nominal_cop(row, subterm_length) for row in raw_data['base_heat_pump_cop_nominal']]
    heat_pump_cop_std = [aggregating_std_cop(row, subterm_length) for row in raw_data['base_heat_pump_cop_std']]
    heat_pump_cop = [[max(1e-4, nom - epsilon * std) for nom, std in zip(n_row, s_row)] for n_row, s_row in zip(heat_pump_cop_nominal, heat_pump_cop_std)]

    electricity_purchasing_cost = [0.144 for _ in range(numStages*numSubperiods+1)]
    heat_purchasing_cost = [0.0374 for _ in range(numStages*numSubperiods+1)]
    emission_limits = [None for _ in range(numStages*numSubperiods)] + [0]
    budget = [0, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000]

    results_directory = f'Results_{numStages}_{numSubperiods}_{numSubterms}_eps({epsilon})'

    electricity_demand = [None] + [electricity_demand[:numSubterms] for _ in range(numStages * numSubperiods)]
    heat_demand = [None] + [heat_demand[:numSubterms] for _ in range(numStages * numSubperiods)]
    electricity_demand_std = [None] + [electricity_demand_std[:numSubterms] for _ in range(numStages * numSubperiods)]
    heat_demand_std = [None] + [heat_demand_std[:numSubterms] for _ in range(numStages * numSubperiods)]

    data = dict(solar_initial=raw_data['solar_initial'], solar_periodic_generation=solar_periodic_generation, solar_advancements=raw_data['solar_advancements'],
    wind_initial=raw_data['wind_initial'], wind_periodic_generation=wind_periodic_generation, wind_advancements=raw_data['wind_advancements'],
    electricity_storage_initial=raw_data['electricity_storage_initial'], electricity_storage_advancements=raw_data['electricity_storage_advancements'],
    parabolic_trough_initial=raw_data['parabolic_trough_initial'], parabolic_trough_periodic_generation=parabolic_trough_periodic_generation,
    parabolic_trough_advancements=raw_data['parabolic_trough_advancements'], heat_pump_initial=raw_data['heat_pump_initial'], heat_pump_cop=heat_pump_cop,
    heat_pump_advancements=raw_data['heat_pump_advancements'], heat_storage_initial=raw_data['heat_storage_initial'], heat_storage_advancements=raw_data['heat_storage_advancements'],
    emission_limits=emission_limits, electricity_demand=electricity_demand, heat_demand=heat_demand, budget=budget,
    electricity_purchasing_cost=electricity_purchasing_cost, heat_purchasing_cost=heat_purchasing_cost,
    results_directory=results_directory, discount_factor=discount_factor,
    electricity_demand_std=electricity_demand_std, heat_demand_std=heat_demand_std,
    solar_periodic_generation_std=solar_periodic_generation_std, wind_periodic_generation_std=wind_periodic_generation_std,
    parabolic_trough_periodic_generation_std=parabolic_trough_periodic_generation_std, heat_pump_cop_std=heat_pump_cop_std)

    return data