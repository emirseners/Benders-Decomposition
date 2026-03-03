import os
import time
from fetch_data import fetch_data
from benders import CampusApplication
from mssp_model import run_mssp_verification
from obtain_incumbent import obtain_incumbent
from scenario_tree import generate_scenario_tree, extract_stage_node_ranges, extract_scenario_paths_and_probabilities

if __name__ == '__main__':
    execution_start_time = time.time()

    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2
    benders_without_feasibility_flag = False
    valid_inequalities_flag = False
    worst_sp_incumbent_flag = False
    continuous_flag = False
    multi_cut_flag = True
    callback_flag = False
    write_cuts_flag = False
    master_threads = 3
    threads_per_worker = 1

    tolerance = 0.01

    input_data = fetch_data(numStages, numSubperiods, numSubterms, epsilon = 0)

    scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, benders_without_feasibility_flag)
    stage_node_ranges = extract_stage_node_ranges(scenario_tree)
    scenario_paths, scenario_path_probabilities = extract_scenario_paths_and_probabilities(scenario_tree)

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

    run_mssp_verification(numStages, numSubperiods, numSubterms, numMultipliers, tolerance)