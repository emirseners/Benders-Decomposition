import os
import time
from fetch_data import fetch_data
from scenario_tree import generate_scenario_tree
from benders import CampusApplication
from obtain_incumbent import obtain_incumbent

if __name__ == '__main__':
    execution_start_time = time.time()

    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2
    number_of_technologies_with_multipliers = 2
    benders_without_feasibility_flag = False
    valid_inequalities_flag = True
    worst_sp_incumbent_flag = False
    continuous_flag = True
    multi_cut_flag = True
    callback_flag = False
    write_cuts_flag = True
    master_threads = 3
    threads_per_worker = 1

    tolerance = 0.01

    input_data = fetch_data(numStages, numSubperiods, numSubterms)

    number_of_branches = numMultipliers ** number_of_technologies_with_multipliers

    stage_node_ranges = {}
    last_node_id = 1
    stage_node_ranges[0] = [0]
    stage_node_ranges[1] = [1]

    for stage in range(2, numStages + 1):
        start_node = last_node_id + 1
        end_node = last_node_id + number_of_branches ** (stage - 1)
        stage_node_ranges[stage] = list(range(start_node, end_node + 1))
        last_node_id = end_node

    number_of_scenario_paths = number_of_branches ** (numStages - 1)
    scenario_paths = {scenario_path_number: [0, 1] for scenario_path_number in range(1, number_of_scenario_paths + 1)}

    for stage in range(2, numStages + 1):
        nodes_in_stage = stage_node_ranges[stage]
        
        scenarios_per_node = number_of_scenario_paths // len(nodes_in_stage)
        
        for scenario_num in range(1, number_of_scenario_paths + 1):
            node_index = (scenario_num - 1) // scenarios_per_node
            node_index = min(node_index, len(nodes_in_stage) - 1)
            
            assigned_node = nodes_in_stage[node_index]
            scenario_paths[scenario_num].append(assigned_node)

    scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, benders_without_feasibility_flag)

    scenario_path_probabilities = {int(each_node.id - sum([number_of_branches ** i for i in range(numStages - 1)])) : each_node.probability  for each_node in scenario_tree.nodes if len(each_node.children) == 0}

    os.makedirs(input_data['results_directory'], exist_ok=True)
    log_file = open(os.path.join(input_data['results_directory'], 'BendersLog.txt'), 'w')

    incumbent_solution = None
    if worst_sp_incumbent_flag:
        incumbent_solution = obtain_incumbent(numStages, numSubperiods, numSubterms, numMultipliers, input_data, stage_node_ranges, benders_without_feasibility_flag, tolerance)

    CampusApplication(numStages, numSubperiods, numSubterms, scenario_tree, initial_tech, input_data['emission_limits'], input_data['electricity_demand'],
                      input_data['heat_demand'], input_data['budget'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['results_directory'], 
                      log_file, input_data['discount_factor'], scenario_paths, scenario_path_probabilities, tolerance, benders_without_feasibility_flag,
                      multi_cut_flag, callback_flag, write_cuts_flag, continuous_flag, valid_inequalities_flag, master_threads, threads_per_worker, incumbent_solution)
    
    summary_lines = [f"Total Time: {time.time() - execution_start_time:.2f} seconds"]
    log_file.write('\n'.join(summary_lines) + '\n')
    log_file.close()