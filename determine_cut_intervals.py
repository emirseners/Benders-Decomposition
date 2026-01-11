from fetch_data import fetch_data
import os
import json

numStages = 3
numSubperiods = 5
numSubterms = 1092
numMultipliers = 2
number_of_technologies_with_multipliers = 2

input_data = fetch_data(numStages, numSubperiods, numSubterms)

results_sol_path = os.path.join(input_data['results_directory'], 'Results.sol')

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

solution_values = {}
with open(results_sol_path, 'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split()
        if len(parts) == 2:
            var_name = parts[0]
            var_value = float(parts[1])
            solution_values[var_name] = var_value

all_cut_intervals = {}

for scenario_path_id, scenario_nodes in scenario_paths.items():
    leaf_node = scenario_nodes[-1]
    cut_intervals = []

    heatcarry_values = []
    electricitycarry_values = []

    for each_subterm in range(1, numSubterms+1):
        heatcarry_values.append(solution_values[f'heatcarry_{leaf_node}[{numStages*numSubperiods},{each_subterm}]'])
        electricitycarry_values.append(solution_values[f'electricitycarry_{leaf_node}[{numStages*numSubperiods},{each_subterm}]'])
    
    if heatcarry_values:
        max_heatcarry = max(heatcarry_values)
        i = 0
        while i < numSubterms:
            if heatcarry_values[i] == max_heatcarry:
                last_max_subterm = i + 1
                while i + 1 < numSubterms and heatcarry_values[i + 1] == max_heatcarry:
                    i += 1
                    last_max_subterm = i + 1
                
                for next_subterm in range(last_max_subterm, numSubterms + 1):
                    if next_subterm == numSubterms + 1 or heatcarry_values[next_subterm - 1] == 0:
                        cut_intervals.append(('heatcarry', last_max_subterm, next_subterm if next_subterm <= numSubterms else numSubterms))
                        break
            i += 1
    
    if electricitycarry_values:
        max_electricitycarry = max(electricitycarry_values)
        i = 0
        while i < numSubterms:
            if electricitycarry_values[i] == max_electricitycarry:
                last_max_subterm = i + 1
                while i + 1 < numSubterms and electricitycarry_values[i + 1] == max_electricitycarry:
                    i += 1
                    last_max_subterm = i + 1
                
                for next_subterm in range(last_max_subterm, numSubterms + 1):
                    if next_subterm == numSubterms + 1 or electricitycarry_values[next_subterm - 1] == 0:
                        cut_intervals.append(('electricitycarry', last_max_subterm, next_subterm if next_subterm <= numSubterms else numSubterms))
                        break
            i += 1
    
    all_cut_intervals[leaf_node] = cut_intervals

cut_intervals_file_path = os.path.join(input_data['results_directory'], 'cut_intervals.json')
with open(cut_intervals_file_path, 'w') as f:
    serializable_intervals = {str(leaf_node): cut_intervals for leaf_node, cut_intervals in all_cut_intervals.items()}
    json.dump(serializable_intervals, f, indent=2)

print(all_cut_intervals)