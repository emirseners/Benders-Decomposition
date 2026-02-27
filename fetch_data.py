import os
import numpy as np
import pandas as pd

def clustering_n_consecutive_data_points(values, n):
    result = []
    for i in range(0, len(values), n):
        cluster_sum = sum(values[i:i+n])
        result.append(cluster_sum)
    return result

def clustering_n_consecutive_data_points_cop(values, n):
    result = []
    for i in range(0, len(values), n):
        cluster_mean = (sum(values[i:i+n]))/n
        result.append(cluster_mean)
    return result

def _read_sheet(sheet_name):
    df = pd.read_excel(os.path.join("Data", "Robust Subterm Data.xlsx"), sheet_name=sheet_name)
    data = df.fillna(0.0)

    means = data.mean(axis=1).tolist()
    half_ranges = ((data.max(axis=1) - data.min(axis=1)) / 2.0).tolist()

    return means, half_ranges

def obtain_robust_data(alpha):
    solar_means, solar_hr = _read_sheet("solar")
    wind_means, wind_hr = _read_sheet("wind")

    base_solar_generation_data = [max(0, m - alpha * hr) for m, hr in zip(solar_means, solar_hr)][:(8760-24)]
    base_wind_generation_data = [max(0, m - alpha * hr) for m, hr in zip(wind_means, wind_hr)][:(8760-24)]

    return [base_solar_generation_data, [2*x for x in base_solar_generation_data]], [base_wind_generation_data]

def _read_std_data(sheet_name):
    daily_std = pd.read_excel(os.path.join('Data', 'Operational Data Daily Std.xlsx'), sheet_name=sheet_name).fillna(0.0)
    cluster_std = pd.read_excel(os.path.join('Data', 'Operational Data Cluster Std.xlsx'), sheet_name=sheet_name).fillna(0.0)
    return daily_std, cluster_std

def fetch_data(numStages, numSubperiods, numSubterms, epsilon = 0):
    discount_factor = 0.97

    electricity_demand_2023 = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='2023 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
    electricity_demand_2023 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2023[j] for j in range(i - 1, -1, -1) if electricity_demand_2023[j] >= 100), None), next((electricity_demand_2023[j] for j in range(i + 1, len(electricity_demand_2023)) if electricity_demand_2023[j] >= 100), None))) for i, x in enumerate(electricity_demand_2023)]
    electricity_demand_2024 = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='2024 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
    electricity_demand_2024 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2024[j] for j in range(i - 1, -1, -1) if electricity_demand_2024[j] >= 100), None), next((electricity_demand_2024[j] for j in range(i + 1, len(electricity_demand_2024)) if electricity_demand_2024[j] >= 100), None))) for i, x in enumerate(electricity_demand_2024)]

    daily_std_2023, cluster_std_2023 = _read_std_data('2023 Hourly Electricity Demand')
    sigma_d_2023 = daily_std_2023.iloc[:, 0].values
    sigma_c_2023 = cluster_std_2023.iloc[:, 0].values
    electricity_demand_2023 = [x + epsilon * sigma_d_2023[i] * sigma_c_2023[i] for i, x in enumerate(electricity_demand_2023)]

    daily_std_2024, cluster_std_2024 = _read_std_data('2024 Hourly Electricity Demand')
    sigma_d_2024 = daily_std_2024.iloc[:, 0].values
    sigma_c_2024 = cluster_std_2024.iloc[:, 0].values
    electricity_demand_2024 = [x + epsilon * sigma_d_2024[i] * sigma_c_2024[i] for i, x in enumerate(electricity_demand_2024)]

    electricity_demand_2023 = electricity_demand_2023[24:]
    electricity_demand_2024 = electricity_demand_2024[:(8760-24)]
    base_electricity_demand = [(val_2023 + val_2024) / 2 for val_2023, val_2024 in zip(electricity_demand_2023, electricity_demand_2024)]

    heat_demand_2024 = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='2024 Hourly Heat Demand')['Consumption (kWh/h)'].tolist()
    heat_demand_2024 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((heat_demand_2024[j] for j in range(i - 1, -1, -1) if heat_demand_2024[j] >= 100), None), next((heat_demand_2024[j] for j in range(i + 1, len(heat_demand_2024)) if heat_demand_2024[j] >= 100), None))) for i, x in enumerate(heat_demand_2024)]

    daily_std_heat, cluster_std_heat = _read_std_data('2024 Hourly Heat Demand')
    sigma_d_heat = daily_std_heat.iloc[:, 0].values
    sigma_c_heat = cluster_std_heat.iloc[:, 0].values
    heat_demand_2024 = [x + epsilon * sigma_d_heat[i] * sigma_c_heat[i] for i, x in enumerate(heat_demand_2024)]
    base_heat_demand = heat_demand_2024[:(8760-24)]

    solar_initial = pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Initial values')
    solar_advancements = {1: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements1'),
                          2: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements2'),
                          3: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements3')}
    nominal_solar = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='solar')
    daily_std_solar, cluster_std_solar = _read_std_data('solar')
    adjusted_solar = nominal_solar.values - epsilon * daily_std_solar.values * cluster_std_solar.values
    adjusted_solar = np.maximum(adjusted_solar, 0)
    base_solar_periodic_generation = [row[:(8760-24)] for row in adjusted_solar.T.tolist()]

    wind_initial = pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Initial values')
    wind_advancements = {1: pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Advancements1')}
    nominal_wind = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='wind')
    daily_std_wind, cluster_std_wind = _read_std_data('wind')
    adjusted_wind = nominal_wind.values - epsilon * daily_std_wind.values * cluster_std_wind.values
    adjusted_wind = np.maximum(adjusted_wind, 0)
    base_wind_periodic_generation = [row[:(8760-24)] for row in adjusted_wind.T.tolist()]

    electricity_storage_initial = pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Initial values')
    electricity_storage_advancements = {1: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements1'),
                                        2: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements2'),
                                        3: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements3')}

    parabolic_trough_initial = pd.read_excel(os.path.join('Data', 'Parabolic Trough.xlsx'), sheet_name='Initial values')
    parabolic_trough_advancements = {1: pd.read_excel(os.path.join('Data', 'Parabolic Trough.xlsx'), sheet_name='Advancements1')}
    nominal_pt = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='parabolic trough')
    daily_std_pt, cluster_std_pt = _read_std_data('parabolic trough')
    adjusted_pt = nominal_pt.values - epsilon * daily_std_pt.values * cluster_std_pt.values
    adjusted_pt = np.maximum(adjusted_pt, 0)
    base_parabolic_trough_periodic_generation = [row[:(8760-24)] for row in adjusted_pt.T.tolist()]

    heat_pump_initial = pd.read_excel(os.path.join('Data', 'Heat Pump.xlsx'), sheet_name='Initial values')
    heat_pump_advancements = {1: pd.read_excel(os.path.join('Data', 'Heat Pump.xlsx'), sheet_name='Advancements1')}
    nominal_hp = pd.read_excel(os.path.join('Data', 'Operational Data Nominal.xlsx'), sheet_name='heat pump')
    daily_std_hp, cluster_std_hp = _read_std_data('heat pump')
    adjusted_hp = nominal_hp.values - epsilon * daily_std_hp.values * cluster_std_hp.values
    adjusted_hp = np.maximum(adjusted_hp, 0)
    base_heat_pump_cop = [row[:(8760-24)] for row in adjusted_hp.T.tolist()]

    heat_storage_initial = pd.read_excel(os.path.join('Data', 'Heat Storage.xlsx'), sheet_name='Initial values')
    heat_storage_advancements = {1: pd.read_excel(os.path.join('Data', 'Heat Storage.xlsx'), sheet_name='Advancements1')}

    electricity_demand = clustering_n_consecutive_data_points(base_electricity_demand, int((8760-24)/numSubterms)).copy()
    heat_demand = clustering_n_consecutive_data_points(base_heat_demand, int((8760-24)/numSubterms)).copy()
    solar_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_solar_periodic_generation].copy()
    wind_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_wind_periodic_generation].copy()
    parabolic_trough_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_parabolic_trough_periodic_generation].copy()
    heat_pump_cop = [clustering_n_consecutive_data_points_cop(version_cop, int((8760-24)/numSubterms)) for version_cop in base_heat_pump_cop].copy()

    electricity_purchasing_cost = [0.144 for _ in range(numStages*numSubperiods+1)]
    heat_purchasing_cost = [0.0374 for _ in range(numStages*numSubperiods+1)]
    emission_limits = [None for _ in range(numStages*numSubperiods)] + [0]
    budget = [0, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000, 30000000]
    #budget = [0] + [None for _ in range(15)]

    results_directory = f'Results_{numStages}_{numSubperiods}_{numSubterms}'
    
    electricity_demand = [None] + [electricity_demand[:numSubterms] for _ in range(numStages * numSubperiods)]
    heat_demand = [None] + [heat_demand[:numSubterms] for _ in range(numStages * numSubperiods)]

    data = dict(solar_initial=solar_initial, solar_periodic_generation=solar_periodic_generation, solar_advancements=solar_advancements,
    wind_initial=wind_initial, wind_periodic_generation=wind_periodic_generation, wind_advancements=wind_advancements,
    electricity_storage_initial=electricity_storage_initial, electricity_storage_advancements=electricity_storage_advancements,
    parabolic_trough_initial=parabolic_trough_initial, parabolic_trough_periodic_generation=parabolic_trough_periodic_generation, 
    parabolic_trough_advancements=parabolic_trough_advancements, heat_pump_initial=heat_pump_initial, heat_pump_cop=heat_pump_cop, 
    heat_pump_advancements=heat_pump_advancements, heat_storage_initial=heat_storage_initial, heat_storage_advancements=heat_storage_advancements, 
    emission_limits=emission_limits, electricity_demand=electricity_demand, heat_demand=heat_demand, budget=budget,
    electricity_purchasing_cost=electricity_purchasing_cost, heat_purchasing_cost=heat_purchasing_cost,
    results_directory=results_directory, discount_factor=discount_factor)

    return data