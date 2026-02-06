import pandas as pd
import os

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

def fetch_data(numStages, numSubperiods, numSubterms):
    discount_factor = 0.97

    electricity_demand_2023 = pd.read_excel(os.path.join('Data', 'Demand.xlsx'), sheet_name='2023 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
    electricity_demand_2023 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2023[j] for j in range(i - 1, -1, -1) if electricity_demand_2023[j] >= 100), None), next((electricity_demand_2023[j] for j in range(i + 1, len(electricity_demand_2023)) if electricity_demand_2023[j] >= 100), None))) for i, x in enumerate(electricity_demand_2023)]
    electricity_demand_2024 = pd.read_excel(os.path.join('Data', 'Demand.xlsx'), sheet_name='2024 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
    electricity_demand_2024 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2024[j] for j in range(i - 1, -1, -1) if electricity_demand_2024[j] >= 100), None), next((electricity_demand_2024[j] for j in range(i + 1, len(electricity_demand_2024)) if electricity_demand_2024[j] >= 100), None))) for i, x in enumerate(electricity_demand_2024)]

    electricity_demand_2023 = electricity_demand_2023[24:]
    electricity_demand_2024 = electricity_demand_2024[:(8760-24)]
    base_electricity_demand = [(val_2023 + val_2024) / 2 for val_2023, val_2024 in zip(electricity_demand_2023, electricity_demand_2024)]

    heat_demand_2024 = pd.read_excel(os.path.join('Data', 'Demand.xlsx'), sheet_name='2024 Hourly Heat Demand')['Consumption (kWh/h)'].tolist()
    heat_demand_2024 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((heat_demand_2024[j] for j in range(i - 1, -1, -1) if heat_demand_2024[j] >= 100), None), next((heat_demand_2024[j] for j in range(i + 1, len(heat_demand_2024)) if heat_demand_2024[j] >= 100), None))) for i, x in enumerate(heat_demand_2024)]
    base_heat_demand = heat_demand_2024[:(8760-24)]

    solar_initial = pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Initial values')
    solar_advancements = {1: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements1'),
                          2: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements2'),
                          3: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements3')}
    base_solar_periodic_generation = [row[:(8760-24)] for row in pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='solar').T.values.tolist()]

    wind_initial = pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Initial values')
    wind_advancements = {1: pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Advancements1')}
    base_wind_periodic_generation = [row[:(8760-24)] for row in pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='wind').T.values.tolist()]

    electricity_storage_initial = pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Initial values')
    electricity_storage_advancements = {1: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements1'),
                                        2: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements2'),
                                        3: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements3')}

    parabolic_trough_initial = pd.read_excel(os.path.join('Data', 'Parabolic Trough.xlsx'), sheet_name='Initial values')
    parabolic_trough_advancements = {1: pd.read_excel(os.path.join('Data', 'Parabolic Trough.xlsx'), sheet_name='Advancements1')}
    base_parabolic_trough_periodic_generation = [row[:(8760-24)] for row in pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='parabolic trough').T.values.tolist()]

    heat_pump_initial = pd.read_excel(os.path.join('Data', 'Heat Pump.xlsx'), sheet_name='Initial values')
    heat_pump_advancements = {1: pd.read_excel(os.path.join('Data', 'Heat Pump.xlsx'), sheet_name='Advancements1')}
    base_heat_pump_cop = [row[:(8760-24)] for row in pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='heat pump').T.values.tolist()]

    heat_storage_initial = pd.read_excel(os.path.join('Data', 'Heat Storage.xlsx'), sheet_name='Initial values')
    heat_storage_advancements = {1: pd.read_excel(os.path.join('Data', 'Heat Storage.xlsx'), sheet_name='Advancements1')}

    #base_solar_periodic_generation, base_wind_periodic_generation = obtain_robust_data(0)

    electricity_demand = clustering_n_consecutive_data_points(base_electricity_demand, int((8760-24)/numSubterms)).copy()
    heat_demand = clustering_n_consecutive_data_points(base_heat_demand, int((8760-24)/numSubterms)).copy()
    solar_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_solar_periodic_generation].copy()
    wind_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_wind_periodic_generation].copy()
    parabolic_trough_periodic_generation = [clustering_n_consecutive_data_points(version_generation, int((8760-24)/numSubterms)) for version_generation in base_parabolic_trough_periodic_generation].copy()
    heat_pump_cop = [clustering_n_consecutive_data_points_cop(version_cop, int((8760-24)/numSubterms)) for version_cop in base_heat_pump_cop].copy()

    electricity_purchasing_cost = [0.144 for _ in range(numStages*numSubperiods+1)]
    heat_purchasing_cost = [0.0374 for _ in range(numStages*numSubperiods+1)]
    emission_limits = [None for _ in range(numStages*numSubperiods)] + [0]
    budget = [0, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000, 20000000]
    #budget = [0] + [None for _ in range(15)]

    results_directory = f'Results_{numStages}_{numSubperiods}_{numSubterms}'
    
    electricity_demand = [electricity_demand[:numSubterms]]*(numStages*numSubperiods + 1)
    heat_demand = [heat_demand[:numSubterms]]*(numStages*numSubperiods + 1)

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