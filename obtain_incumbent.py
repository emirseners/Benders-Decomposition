from mssp_model import MSSPProblemModel
from scenario_tree import generate_scenario_tree
import pandas as pd
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
        numSubterms, numSubperiods, numStages, 1, mssp_flag=True)
    
    worst_sp_model = MSSPProblemModel(worst_scenario_path_scenario_tree, input_data['emission_limits'], input_data['electricity_demand'], 
                        input_data['heat_demand'], worst_scenario_path_initial_tech, input_data['budget'], input_data['electricity_purchasing_cost'],
                        input_data['heat_purchasing_cost'], input_data['results_directory'], input_data['discount_factor'], None, tolerance, 'WorstIncumbent')
    
    last_subperiod = numStages * numSubperiods
    second_to_last_subperiod = last_subperiod - 1
    
    incumbent_solution = {}
    
    for v in worst_sp_model.getVars():
        dv_name, node_id, indices = parse(v.varName)
        
        for each_node_id in stage_node_ranges[node_id]:
            
            if dv_name == 'plus':
                incumbent_solution[f'plus_{each_node_id}[{indices[0]},{indices[1]},{indices[2]}]'] = v.X
            
            if benders_without_feasibility_flag:

                if node_id == numStages:

                    last_subperiod = numStages * numSubperiods
                    second_to_last_subperiod = last_subperiod - 1

                    subperiod = int(indices[0])
                    
                    if dv_name in ['electricitycharge', 'heatcharge', 'electricitydischarge', 'heatdischarge', 'electricityused', 'heatused']:
                        if subperiod == last_subperiod:
                            incumbent_solution[f'{dv_name}_{each_node_id}[{indices[0]},{indices[1]}]'] = v.X
                    
                    elif dv_name in ['electricitycarry', 'heatcarry']:
                        p = int(indices[1])
                        if subperiod == last_subperiod:
                            incumbent_solution[f'{dv_name}_{each_node_id}[{indices[0]},{indices[1]}]'] = v.X
                        elif subperiod == second_to_last_subperiod and p == numSubterms:
                            incumbent_solution[f'{dv_name}_{each_node_id}[{indices[0]},{indices[1]}]'] = v.X
                    
                    elif dv_name == 'transferredheat':
                        t_ = int(indices[4])
                        if t_ == last_subperiod:
                            incumbent_solution[f'{dv_name}_{each_node_id}[{indices[0]},{indices[1]},{indices[2]},{indices[3]},{indices[4]}]'] = v.X

    return incumbent_solution