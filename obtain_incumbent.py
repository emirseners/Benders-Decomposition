from scenario_tree import generate_scenario_tree, extract_scenario_paths_and_probabilities
from benders import run_benders
import pandas as pd
import os
import re

def parse(s: str):
    m = re.match(r'([a-zA-Z]+)_(\d+)\[([^\]]+)\]', s)

    dv_name = m.group(1).strip()
    node_id = int(m.group(2).strip())
    indices = [p.strip() for p in m.group(3).split(',')]

    return dv_name, node_id, indices

def obtain_incumbent(numStages, numSubperiods, numSubterms, numMultipliers, input_data, stage_node_ranges, benders_without_feasibility_flag, tolerance):

    technology_advancements = {'solar': input_data['solar_advancements'], 'electricity_storage': input_data['electricity_storage_advancements'], 
                               'wind': input_data['wind_advancements'], 'parabolic_trough': input_data['parabolic_trough_advancements'],
                               'heat_pump': input_data['heat_pump_advancements'], 'heat_storage': input_data['heat_storage_advancements']}

    worst_technology_advancements = {}

    for technology_name, each in technology_advancements.items():
        key = numMultipliers if numMultipliers in each else 1
        worst_cost_ratio = min([each[key][col][1] for col in each[key].columns if col != "Metrics"])
        worst_efficiency_ratio = min([each[key][col][2] for col in each[key].columns if col != "Metrics"])
        worst_cost_multiplier = max([each[key][col][3] for col in each[key].columns if col != "Metrics"])
        worst_efficiency_multiplier = min([each[key][col][4] for col in each[key].columns if col != "Metrics"])

        tech_df = pd.DataFrame({
            "Metrics": ["Probabilities", "Cost Ratio", "Efficiency Ratio", "Cost Multiplier", "Efficiency Multiplier", "Emission Multiplier"],
            "Scenario1": [1, worst_cost_ratio, worst_efficiency_ratio, worst_cost_multiplier, worst_efficiency_multiplier, 0]})

        worst_technology_advancements[technology_name] = {1: tech_df}
    
    worst_scenario_path_scenario_tree, worst_scenario_path_initial_tech = generate_scenario_tree(
        input_data['solar_initial'], input_data['solar_periodic_generation'], worst_technology_advancements['solar'],
        input_data['wind_initial'], input_data['wind_periodic_generation'], worst_technology_advancements['wind'],
        input_data['electricity_storage_initial'], worst_technology_advancements['electricity_storage'],
        input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], worst_technology_advancements['parabolic_trough'],
        input_data['heat_pump_initial'], input_data['heat_pump_cop'], worst_technology_advancements['heat_pump'],
        input_data['heat_storage_initial'], worst_technology_advancements['heat_storage'],
        numSubterms, numSubperiods, numStages, 1, benders_without_feasibility_flag)
    
    worst_scenario_paths, worst_scenario_path_probabilities = extract_scenario_paths_and_probabilities(worst_scenario_path_scenario_tree)
    
    log_file = open(os.path.join(input_data['results_directory'], 'IncumbentBendersLog.csv'), 'w', newline='')

    solution = run_benders(numStages, numSubperiods, numSubterms, worst_scenario_path_scenario_tree, worst_scenario_path_initial_tech,
                           input_data['emission_limits'], input_data['electricity_demand'], input_data['heat_demand'], input_data['budget'],
                           input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['results_directory'],
                           log_file, input_data['discount_factor'], worst_scenario_paths, worst_scenario_path_probabilities, tolerance,
                           benders_without_feasibility_flag, False, False, True, False, 4, 1, None, True)

    log_file.close()

    incumbent_solution = {}
    
    for varName, val in solution.items():        
        dv_name, node_id, indices = parse(varName)
        node_stage = worst_scenario_path_scenario_tree.nodes[node_id].stage
        
        for each_node_id in stage_node_ranges[node_stage]:
            if "plus" in dv_name:
                incumbent_solution[f"{dv_name}_{each_node_id}[{','.join(indices)}]"] = val

    return incumbent_solution