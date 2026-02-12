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
    numScenarioPaths = max(scenario_paths)
    numTotalPeriods = int(numStages * numSubperiods)

    operational_data = {i: {'Electricity Generation': [[0 for _ in range(numSubterms)] for _ in range(numTotalPeriods)],
                            'Heat Generation': [[0 for _ in range(numSubterms)] for _ in range(numTotalPeriods)],
                            'Annual Electricity Purchase': [0 for _ in range(numTotalPeriods)],
                            'Annual Heat Purchase': [0 for _ in range(numTotalPeriods)],
                            'Electricity Storage Capacity': [0 for _ in range(numTotalPeriods)],
                            'Heat Storage Capacity': [0 for _ in range(numTotalPeriods)],
                            'Heat Transfer Capacity': [0 for _ in range(numTotalPeriods)]
                            } for i in range(1, numScenarioPaths+1)}
    
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
                t = int(indices[2]) - 1
                max_year = min(numTotalPeriods, t + technology.lifetime[v])

                if technology.tree.segment == 'electricity generation':
                    for t_ in range(t, max_year):
                        for p in range(numSubterms):
                            operational_data[sp_id]['Electricity Generation'][t_][p] += technology.periodic_electricity[v][p] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat generation':
                    for t_ in range(t, max_year):
                        for p in range(numSubterms):
                            operational_data[sp_id]['Heat Generation'][t_][p] += technology.periodic_heat[v][p] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'electricity storage':
                    for t_ in range(t, max_year):
                        operational_data[sp_id]['Electricity Storage Capacity'][t_] += technology.electricity_storage_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat storage':
                    for t_ in range(t, max_year):
                        operational_data[sp_id]['Heat Storage Capacity'][t_] += technology.heat_storage_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

                if technology.tree.segment == 'heat transfer':
                    for t_ in range(t, max_year):
                        operational_data[sp_id]['Heat Transfer Capacity'][t_] += technology.heat_transfer_capacity[v] * (1 - (technology.degradation_rate[v] * (t_ - t))) * dv_value

    heat_cop = scenario_tree.nodes[-1].heattransfertechNodeList[0].periodic_heat_transfer_cop[0]

    return operational_data, heat_cop

def run_simulation(operational_data, heat_cop, scenario_paths, scenario_path_probabilities, numStages, numSubperiods, numSubterms, electricity_demand, heat_demand):
    numScenarioPaths = max(scenario_paths)
    numTotalPeriods = int(numStages * numSubperiods)

    outputs = {i: {'Electricity Violation': [0 for _ in range(numTotalPeriods)],
                   'Heat Violation': [0 for _ in range(numTotalPeriods)]
                   } for i in range(1, numScenarioPaths+1)}

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

            electricity_violation = 0
            heat_violation = 0

            for s in range(numSubterms):
                electricity_demand_at_subperiod = electricity_demand[s]
                heat_demand_at_subperiod = heat_demand[s]
                surplus_electricity = 0

                if electricity_generation[s] >= electricity_demand_at_subperiod:
                    surplus_electricity = electricity_generation[s] - electricity_demand_at_subperiod
                    stored_electricity = min(stored_electricity + (0.9 * (electricity_generation[s] - electricity_demand_at_subperiod)), electricity_storage_capacity)
                
                else:
                    remaining_electricity_demand_at_subperiod = electricity_demand_at_subperiod - electricity_generation[s]

                    if stored_electricity * 0.9 >= remaining_electricity_demand_at_subperiod:
                        stored_electricity -= remaining_electricity_demand_at_subperiod / 0.9

                    else:
                        remaining_electricity_demand_after_electricity_storage = remaining_electricity_demand_at_subperiod - (stored_electricity * 0.9)
                        stored_electricity = 0

                        if annual_electricity_purchase >= remaining_electricity_demand_after_electricity_storage:
                            annual_electricity_purchase -= remaining_electricity_demand_after_electricity_storage

                        else:
                            not_met_electricity_demand = remaining_electricity_demand_after_electricity_storage - annual_electricity_purchase
                            annual_electricity_purchase = 0
                            electricity_violation += not_met_electricity_demand
                
                transferable_heat_from_surplus = min(heat_transfer_capacity, surplus_electricity * heat_cop[s])

                if heat_generation[s] + transferable_heat_from_surplus >= heat_demand_at_subperiod:
                    stored_heat = min(stored_heat + (0.9 * max(0, heat_generation[s] - heat_demand_at_subperiod)), heat_storage_capacity)
                
                else:
                    remaining_heat_demand_at_subperiod = heat_demand_at_subperiod - (heat_generation[s] + transferable_heat_from_surplus)
                    remaining_heat_transfer_capacity = heat_transfer_capacity - transferable_heat_from_surplus
                    remaining_transferable_heat_from_electricity_storage = min(stored_electricity * 0.9 * heat_cop[s], remaining_heat_transfer_capacity)

                    if remaining_transferable_heat_from_electricity_storage >= remaining_heat_demand_at_subperiod:
                        stored_electricity -= remaining_heat_demand_at_subperiod / (0.9 * heat_cop[s])
                    
                    else:
                        stored_electricity -= remaining_transferable_heat_from_electricity_storage / (0.9 * heat_cop[s])
                        remaining_heat_demand_after_electricity_storage = remaining_heat_demand_at_subperiod - remaining_transferable_heat_from_electricity_storage
                        remaining_heat_transfer_capacity -= remaining_transferable_heat_from_electricity_storage

                        if stored_heat * 0.9 >= remaining_heat_demand_after_electricity_storage:
                            stored_heat -= remaining_heat_demand_after_electricity_storage / 0.9
                        
                        else:
                            remaining_heat_demand_after_heat_storage = remaining_heat_demand_after_electricity_storage - (stored_heat * 0.9)
                            stored_heat = 0

                            if heat_cop[s] >= 0.144 / 0.0374:
                                if annual_electricity_purchase >= remaining_heat_demand_after_heat_storage / heat_cop[s]:
                                    heat_transferred = min(remaining_heat_transfer_capacity, remaining_heat_demand_after_heat_storage)
                                    remaining_heat_demand_after_electricity_purchase_transfer = max(remaining_heat_demand_after_heat_storage - heat_transferred, 0)
                                    annual_electricity_purchase -= heat_transferred / heat_cop[s]

                                    if remaining_heat_demand_after_electricity_purchase_transfer > 0:

                                        if annual_heat_purchase >= remaining_heat_demand_after_electricity_purchase_transfer:
                                            annual_heat_purchase -= remaining_heat_demand_after_electricity_purchase_transfer
                                        
                                        else:
                                            not_met_heat_demand = remaining_heat_demand_after_electricity_purchase_transfer - annual_heat_purchase
                                            annual_heat_purchase = 0
                                            heat_violation += not_met_heat_demand
                                else:
                                    remaining_heat_demand_after_electricity_purchase_transfer = remaining_heat_demand_after_heat_storage - min(remaining_heat_transfer_capacity, annual_electricity_purchase * heat_cop[s])
                                    annual_electricity_purchase = 0

                                    if remaining_heat_demand_after_electricity_purchase_transfer > 0:

                                        if annual_heat_purchase >= remaining_heat_demand_after_electricity_purchase_transfer:
                                            annual_heat_purchase -= remaining_heat_demand_after_electricity_purchase_transfer

                                        else:
                                            not_met_heat_demand = remaining_heat_demand_after_electricity_purchase_transfer - annual_heat_purchase
                                            annual_heat_purchase = 0
                                            heat_violation += not_met_heat_demand
                            
                            else:
                                if annual_heat_purchase >= remaining_heat_demand_after_heat_storage:
                                    annual_heat_purchase -= remaining_heat_demand_after_heat_storage
                                else:
                                    remaining_heat_demand_after_heat_purchase = remaining_heat_demand_after_heat_storage - annual_heat_purchase
                                    annual_heat_purchase = 0

                                    heat_transferred = min(remaining_heat_transfer_capacity, annual_electricity_purchase * heat_cop[s])
                                    remaining_heat_demand_after_electricity_purchase_transfer = remaining_heat_demand_after_heat_purchase - heat_transferred
                                    annual_electricity_purchase -= heat_transferred / heat_cop[s]

                                    if remaining_heat_demand_after_electricity_purchase_transfer > 0:
                                        heat_violation += remaining_heat_demand_after_electricity_purchase_transfer

            outputs[sp_id]['Electricity Violation'][t] = electricity_violation
            outputs[sp_id]['Heat Violation'][t] = heat_violation
    
    total_electricity_violation_by_sp = {sp_id: sum(outputs[sp_id]['Electricity Violation']) for sp_id in range(1, numScenarioPaths+1)}
    total_heat_violation_by_sp = {sp_id: sum(outputs[sp_id]['Heat Violation']) for sp_id in range(1, numScenarioPaths+1)}

    total_electricity_violation_cost_by_sp = {sp_id: sum([each_year_demand * 0.144 * (0.97 ** t) for t, each_year_demand in enumerate(outputs[sp_id]['Electricity Violation'])]) for sp_id in range(1, numScenarioPaths+1)}
    total_heat_violation_cost_by_sp = {sp_id: sum([each_year_demand * 0.0374 * (0.97 ** t) for t, each_year_demand in enumerate(outputs[sp_id]['Heat Violation'])]) for sp_id in range(1, numScenarioPaths+1)}

    average_electricity_violation = sum(total_electricity_violation_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_electricity_violation_by_sp)
    average_heat_violation = sum(total_heat_violation_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_heat_violation_by_sp)

    average_electricity_violation_cost = sum(total_electricity_violation_cost_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_electricity_violation_cost_by_sp)
    average_heat_violation_cost = sum(total_heat_violation_cost_by_sp[sp_id] * scenario_path_probabilities[sp_id] for sp_id in total_heat_violation_cost_by_sp)

    return average_electricity_violation, average_heat_violation, average_electricity_violation_cost, average_heat_violation_cost

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

    operational_data, heat_cop = obtain_operational_data(scenario_tree, filtered_results_by_path, numStages, numSubperiods, numSubterms, scenario_paths)

    electricity_demand = input_data['electricity_demand'][-1]
    heat_demand = input_data['heat_demand'][-1]

    results = run_simulation(operational_data, heat_cop, scenario_paths, scenario_path_probabilities, numStages, numSubperiods, numSubterms, electricity_demand, heat_demand)

    print(results)