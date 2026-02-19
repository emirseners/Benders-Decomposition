import os
import re
import random
from fetch_data import fetch_data
from scenario_tree import generate_scenario_tree, extract_stage_node_ranges, extract_scenario_paths_and_probabilities

def read_results_sol(results_directory):
    sol_path = os.path.join(results_directory, 'Results.sol')
    sol_values = {}
    with open(sol_path, 'r') as sol_file:
        for line in sol_file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            sol_values[parts[0]] = float(parts[1])
    return sol_values

def filter_sol_values_by_prefix(sol_values, prefixes):
    return {name: value for name, value in sol_values.items() if name.startswith(prefixes)}

def filter_sol_values_by_nodes(sol_values, node_ids):
    allowed_nodes = {str(node_id) for node_id in node_ids}
    filtered = {}
    for name, value in sol_values.items():
        node_part = name.split('_', 1)[1].split('[', 1)[0]
        if node_part in allowed_nodes:
            filtered[name] = value
    return filtered

def parse(s: str):
    m = re.match(r'([a-zA-Z]+)_(\d+)\[([^\]]+)\]', s)

    dv_name = m.group(1).strip()
    node_id = int(m.group(2).strip())
    indices = [p.strip() for p in m.group(3).split(',')]

    return dv_name, node_id, indices

def obtain_operational_data(scenario_tree, filtered_results_by_path, numStages, numSubperiods, numSubterms, scenario_paths):
    scenario_ids = sorted(scenario_paths.keys())
    numTotalPeriods = int(numStages * numSubperiods)

    operational_data = {i: {'Electricity Generation': [[0 for _ in range(numSubterms)] for _ in range(numTotalPeriods)],
                            'Heat Generation': [[0 for _ in range(numSubterms)] for _ in range(numTotalPeriods)],
                            'Annual Electricity Purchase': [0 for _ in range(numTotalPeriods)],
                            'Annual Heat Purchase': [0 for _ in range(numTotalPeriods)],
                            'Electricity Storage Capacity': [0 for _ in range(numTotalPeriods)],
                            'Heat Storage Capacity': [0 for _ in range(numTotalPeriods)],
                            'Heat Transfer Capacity': [0 for _ in range(numTotalPeriods)],
                            'Effective Heat Transfer Capacity': [0 for _ in range(numTotalPeriods)],
                            'Heat Cop': [[0 for _ in range(numSubterms)] for _ in range(numTotalPeriods)]
                            } for i in scenario_ids}
    
    for sp_id, decision_variables in filtered_results_by_path.items():
        for dv_name, dv_value in decision_variables.items():
            if dv_value == 0:
                continue

            dv_group, node_id, indices = parse(dv_name)

            if dv_group == 'electricitypurchase':
                operational_data[sp_id]['Annual Electricity Purchase'][int(indices[0])-1] += dv_value
            elif dv_group == 'heatpurchase':
                operational_data[sp_id]['Annual Heat Purchase'][int(indices[0])-1] += dv_value
            else:
                node = next(node for node in scenario_tree.nodes if node.id == node_id)
                technology = next(tech for tech in node.techNodeList if tech.tree.type == indices[0])

                v = int(indices[1])
                t = int(indices[2])

                if technology.tree.segment == 'electricity generation':
                    for t_ in range(max(t, 1), min(numTotalPeriods + 1, t + technology.lifetime[v])):
                        for p in range(numSubterms):
                            operational_data[sp_id]['Electricity Generation'][t_-1][p] += technology.periodic_electricity[v][p] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat generation':
                    for t_ in range(max(t, 1), min(numTotalPeriods + 1, t + technology.lifetime[v])):
                        for p in range(numSubterms):
                            operational_data[sp_id]['Heat Generation'][t_-1][p] += technology.periodic_heat[v][p] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'electricity storage':
                    for t_ in range(max(t, 1), min(numTotalPeriods + 1, t + technology.lifetime[v])):
                        operational_data[sp_id]['Electricity Storage Capacity'][t_-1] += technology.electricity_storage_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat storage':
                    for t_ in range(max(t, 1), min(numTotalPeriods + 1, t + technology.lifetime[v])):
                        operational_data[sp_id]['Heat Storage Capacity'][t_-1] += technology.heat_storage_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat transfer':
                    for t_ in range(max(t, 1), min(numTotalPeriods + 1, t + technology.lifetime[v])):
                        operational_data[sp_id]['Heat Transfer Capacity'][t_-1] += technology.heat_transfer_capacity[v] * dv_value
                        operational_data[sp_id]['Effective Heat Transfer Capacity'][t_-1] += technology.heat_transfer_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

    for sp_id, stage_node_ids in scenario_paths.items():
        for t in range(numTotalPeriods):
            stage_index = t // numSubperiods
            node_id = stage_node_ids[stage_index + 1]
            node = next(n for n in scenario_tree.nodes if n.id == node_id)
            operational_data[sp_id]['Heat Cop'][t] = node.heattransfertechNodeList[0].periodic_heat_transfer_cop[0]

    first_node = next(n for n in scenario_tree.nodes if n.id != 0)
    e_charging_efficiency = first_node.electricitystoragetechNodeList[0].storage_charging_efficiency[0]
    e_discharging_efficiency = first_node.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]
    h_charging_efficiency = first_node.heatstoragetechNodeList[0].storage_charging_efficiency[0]
    h_discharging_efficiency = first_node.heatstoragetechNodeList[0].storage_discharging_efficiency[0]
    efficiencies = (e_charging_efficiency, e_discharging_efficiency, h_charging_efficiency, h_discharging_efficiency)

    return operational_data, efficiencies

def run_simulation(operational_data, efficiencies, scenario_paths, scenario_path_probabilities, numStages, numSubperiods, numSubterms, electricity_demand, heat_demand):
    scenario_ids = sorted(scenario_paths.keys())
    numTotalPeriods = int(numStages * numSubperiods)
    e_charging_efficiency, e_discharging_efficiency, h_charging_efficiency, h_discharging_efficiency = efficiencies

    outputs = {i: {'Electricity Violation': [0 for _ in range(numTotalPeriods)],
                   'Heat Violation': [0 for _ in range(numTotalPeriods)]
                   } for i in scenario_ids}

    for sp_id, sp_operational_data in operational_data.items():
        stored_electricity = 0
        stored_heat = 0

        for t in range(numTotalPeriods):
            electricity_generation = sp_operational_data['Electricity Generation'][t]
            heat_generation = sp_operational_data['Heat Generation'][t]
            annual_electricity_purchase = sp_operational_data['Annual Electricity Purchase'][t]
            annual_heat_purchase = sp_operational_data['Annual Heat Purchase'][t]
            electricity_storage_capacity = sp_operational_data['Electricity Storage Capacity'][t]
            heat_storage_capacity = sp_operational_data['Heat Storage Capacity'][t]
            heat_transfer_capacity = sp_operational_data['Heat Transfer Capacity'][t]
            effective_heat_transfer_capacity = sp_operational_data['Effective Heat Transfer Capacity'][t]
            heat_cop = sp_operational_data['Heat Cop'][t]

            stored_electricity = min((0.9994 ** (8736/numSubterms)) * stored_electricity, electricity_storage_capacity)
            stored_heat = min((0.995 ** (8736/numSubterms)) * stored_heat, heat_storage_capacity)

            degradation_factor = effective_heat_transfer_capacity / heat_transfer_capacity if heat_transfer_capacity > 0 else 1.0
            
            pooled_electricity_purchase = 0
            pooled_heat_purchase = 0

            electricity_violation = 0
            heat_violation = 0

            for s in range(numSubterms):
                electricity_demand_at_subperiod = electricity_demand[t][s]
                heat_demand_at_subperiod = heat_demand[t][s]
                surplus_electricity = 0
                storable_electricity = 0

                if electricity_generation[s] >= electricity_demand_at_subperiod:
                    max_storable = (electricity_storage_capacity - stored_electricity) / e_charging_efficiency
                    excess = electricity_generation[s] - electricity_demand_at_subperiod
                    if excess <= max_storable:
                        storable_electricity = excess
                    else:
                        surplus_electricity = excess - max_storable
                        storable_electricity = max_storable

                else:
                    remaining_electricity_demand = electricity_demand_at_subperiod - electricity_generation[s]

                    if stored_electricity * e_discharging_efficiency >= remaining_electricity_demand:
                        stored_electricity -= remaining_electricity_demand / e_discharging_efficiency
                    else:
                        remaining_after_e_storage = remaining_electricity_demand - (stored_electricity * e_discharging_efficiency)
                        stored_electricity = 0
                        pooled_electricity_purchase += remaining_after_e_storage

                effective_cop = degradation_factor * heat_cop[s]
                remaining_effective_htc = effective_heat_transfer_capacity
                remaining_heat_demand = heat_demand_at_subperiod

                transferable_from_surplus = min(remaining_heat_demand, remaining_effective_htc, surplus_electricity * effective_cop)
                remaining_effective_htc -= transferable_from_surplus
                remaining_heat_demand -= transferable_from_surplus

                max_storable_heat = (heat_storage_capacity - stored_heat) / h_charging_efficiency
                if heat_generation[s] > max_storable_heat:
                    surplus_heat = heat_generation[s] - max_storable_heat
                    storable_heat_gen = max_storable_heat
                else:
                    surplus_heat = 0
                    storable_heat_gen = heat_generation[s]

                if remaining_heat_demand > 0 and surplus_heat > 0:
                    heat_from_surplus = min(remaining_heat_demand, surplus_heat)
                    remaining_heat_demand -= heat_from_surplus

                if effective_cop >= 0.144 / 0.0374:

                    # Electricity storage
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, stored_electricity * e_discharging_efficiency * effective_cop, remaining_effective_htc)
                        stored_electricity -= transferable / (e_discharging_efficiency * effective_cop)
                        stored_electricity = max(0, stored_electricity)
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                    # Heat generation
                    heat_gen_for_demand = min(storable_heat_gen, remaining_heat_demand)
                    remaining_heat_demand -= heat_gen_for_demand
                    remaining_heat_gen = storable_heat_gen - heat_gen_for_demand

                    # Heat storage
                    if remaining_heat_demand > 0:
                        from_h_storage = min(remaining_heat_demand, stored_heat * h_discharging_efficiency)
                        stored_heat -= from_h_storage / h_discharging_efficiency
                        remaining_heat_demand -= from_h_storage

                    # Transfer storable electricity
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, storable_electricity * effective_cop, remaining_effective_htc)
                        storable_electricity -= transferable / effective_cop
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                    # Electricity purchase
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, remaining_effective_htc)
                        needed_elec = transferable / effective_cop
                        pooled_electricity_purchase += needed_elec
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                    # Heat purchase
                    if remaining_heat_demand > 0:
                        pooled_heat_purchase += remaining_heat_demand
                        remaining_heat_demand = 0

                else:
                    # Electricity storage
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, stored_electricity * e_discharging_efficiency * effective_cop, remaining_effective_htc)
                        stored_electricity -= transferable / (e_discharging_efficiency * effective_cop)
                        stored_electricity = max(0, stored_electricity)
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                    # Heat generation
                    heat_gen_for_demand = min(storable_heat_gen, remaining_heat_demand)
                    remaining_heat_demand -= heat_gen_for_demand
                    remaining_heat_gen = storable_heat_gen - heat_gen_for_demand

                    # Heat storage
                    if remaining_heat_demand > 0:
                        from_h_storage = min(remaining_heat_demand, stored_heat * h_discharging_efficiency)
                        stored_heat -= from_h_storage / h_discharging_efficiency
                        remaining_heat_demand -= from_h_storage

                    # Heat purchase
                    if remaining_heat_demand > 0:
                        pooled_heat_purchase += remaining_heat_demand
                        remaining_heat_demand = 0

                    # Transfer storable electricity
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, storable_electricity * effective_cop, remaining_effective_htc)
                        storable_electricity -= transferable / effective_cop
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                    # Electricity purchase transfer
                    if remaining_heat_demand > 0 and remaining_effective_htc > 0:
                        transferable = min(remaining_heat_demand, remaining_effective_htc)
                        needed_elec = transferable / effective_cop
                        pooled_electricity_purchase += needed_elec
                        remaining_heat_demand -= transferable
                        remaining_effective_htc -= transferable

                stored_heat = min(stored_heat + h_charging_efficiency * remaining_heat_gen, heat_storage_capacity)
                stored_electricity = min(stored_electricity + e_charging_efficiency * storable_electricity, electricity_storage_capacity)

            if annual_electricity_purchase > pooled_electricity_purchase:
                remaining_budget = annual_electricity_purchase - pooled_electricity_purchase
                stored_electricity = min(stored_electricity + e_charging_efficiency * remaining_budget, electricity_storage_capacity)
            else:
                electricity_violation += (pooled_electricity_purchase - annual_electricity_purchase)

            if annual_heat_purchase > pooled_heat_purchase:
                remaining_budget = annual_heat_purchase - pooled_heat_purchase
                stored_heat = min(stored_heat + h_charging_efficiency * remaining_budget, heat_storage_capacity)
            else:
                heat_violation += (pooled_heat_purchase - annual_heat_purchase)

            outputs[sp_id]['Electricity Violation'][t] = electricity_violation
            outputs[sp_id]['Heat Violation'][t] = heat_violation

    total_electricity_violation_by_sp = {sp_id: sum(outputs[sp_id]['Electricity Violation']) for sp_id in scenario_ids}
    total_heat_violation_by_sp = {sp_id: sum(outputs[sp_id]['Heat Violation']) for sp_id in scenario_ids}

    total_electricity_violation_cost_by_sp = {sp_id: sum([each_year_demand * 0.144 * (0.97 ** t) for t, each_year_demand in enumerate(outputs[sp_id]['Electricity Violation'])]) for sp_id in scenario_ids}
    total_heat_violation_cost_by_sp = {sp_id: sum([each_year_demand * 0.0374 * (0.97 ** t) for t, each_year_demand in enumerate(outputs[sp_id]['Heat Violation'])]) for sp_id in scenario_ids}

    average_electricity_violation = sum(total_electricity_violation_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_electricity_violation_by_sp)
    average_heat_violation = sum(total_heat_violation_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_heat_violation_by_sp)

    average_electricity_violation_cost = sum(total_electricity_violation_cost_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_electricity_violation_cost_by_sp)
    average_heat_violation_cost = sum(total_heat_violation_cost_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_heat_violation_cost_by_sp)

    return average_electricity_violation, average_heat_violation, average_electricity_violation_cost, average_heat_violation_cost, average_electricity_violation_cost + average_heat_violation_cost

if __name__ == '__main__':
    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2

    input_data = fetch_data(numStages, numSubperiods, numSubterms)
    optimization_results = read_results_sol(input_data['results_directory'])

    scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, mssp_flag=True)
    stage_node_ranges = extract_stage_node_ranges(scenario_tree)
    scenario_paths, scenario_path_probabilities = extract_scenario_paths_and_probabilities(scenario_tree)

    optimization_results = filter_sol_values_by_prefix(optimization_results, ('electricitypurchase_', 'heatpurchase_', 'plus_'))

    filtered_results_by_path = {}
    for sp_id, stage_node_ids in scenario_paths.items():
        filtered_results_by_path[sp_id] = filter_sol_values_by_nodes(optimization_results, stage_node_ids)

    electricity_demand = input_data['electricity_demand'][1:]
    heat_demand = input_data['heat_demand'][1:]

    operational_data, efficiencies = obtain_operational_data(scenario_tree, filtered_results_by_path, numStages, numSubperiods, numSubterms, scenario_paths)

    results = run_simulation(operational_data, efficiencies, scenario_paths, scenario_path_probabilities, numStages, numSubperiods, numSubterms, electricity_demand, heat_demand)

    print(results)