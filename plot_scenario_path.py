import os
import re
import pandas as pd
import plotly.io as pio
import plotly.express as px
from fetch_data import fetch_data
from scenario_tree import generate_scenario_tree, extract_scenario_paths_and_probabilities

pio.renderers.default = "browser"

def parse_var_name(name):
	m = re.match(r'([^_]+)_([^\[]+)\[([^\]]+)\]', name)
	if not m:
		return None, None, []
	return m.group(1).strip(), m.group(2).strip(), [p.strip() for p in m.group(3).split(',')]

def load_op_variables_df(results_directory):
	sol_filename = os.path.join(results_directory, "Results.sol")

	rows = []
	with open(sol_filename, "r") as file_handle:
		for line in file_handle:
			line = line.strip()
			if not line or line.startswith("#"):
				continue

			if line.startswith("plus"):
				continue

			parts = line.split()
			if len(parts) < 2:
				continue

			var_name = parts[0]
			value = float(parts[1])

			dv_name, node_id, indices = parse_var_name(var_name)
			if not dv_name:
				continue

			if dv_name == "transferredheat":
				year = indices[4]
				subterm = indices[0]
			else:
				year = indices[0]
				subterm = indices[1]

			rows.append({
				"var": dv_name,
				"node": int(node_id),
				"year": int(year),
				"subterm": int(subterm),
				"value": value,
			})

	df = pd.DataFrame(rows)
	df = df.groupby(["var", "node", "year", "subterm"], as_index=False)["value"].sum()
	return df

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
							} for i in scenario_ids}

	for sp_id, decision_variables in filtered_results_by_path.items():
		for dv_name, dv_value in decision_variables.items():
			if dv_value == 0:
				continue

			dv_group, node_id, indices = parse(dv_name)

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

	return operational_data

def compute_generation_data(numStages, numSubperiods, numSubterms, numMultipliers, epsilon, folder_suffix):
	input_data = fetch_data(numStages, numSubperiods, numSubterms, epsilon=epsilon, folder_suffix=folder_suffix)
	optimization_results = read_results_sol(input_data["results_directory"])
	
	scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, mssp_flag=True)

	scenario_paths, _ = extract_scenario_paths_and_probabilities(scenario_tree)

	optimization_results = filter_sol_values_by_prefix(optimization_results, "plus_")

	filtered_results_by_path = {}
	for sp_id, stage_node_ids in scenario_paths.items():
		filtered_results_by_path[sp_id] = filter_sol_values_by_nodes(optimization_results, stage_node_ids)

	operational_data = obtain_operational_data(scenario_tree, filtered_results_by_path, numStages, numSubperiods, numSubterms, scenario_paths)

	electricity_demand = input_data['electricity_demand'][1:]
	heat_demand = input_data['heat_demand'][1:]

	return operational_data, scenario_paths, electricity_demand, heat_demand, input_data["results_directory"]

def main():
	numStages = 3
	numSubperiods = 5
	numSubterms = 1092
	numMultipliers = 2
	epsilon = 0
	folder_suffix = f"eps({epsilon})"

	operational_data, scenario_paths, electricity_demand, heat_demand, results_dir = compute_generation_data(numStages, numSubperiods, numSubterms, numMultipliers, epsilon, folder_suffix)
	first_sp_id = min(scenario_paths.keys())
	sp_data = operational_data[first_sp_id]
	numTotalPeriods = numStages * numSubperiods

	nodes = [n for n in scenario_paths[first_sp_id] if n != 0]

	df = load_op_variables_df(results_dir)

	exclude_vars = {"electricityused", "heatused"}

	df_filtered = df[df["node"].isin(nodes) & ~df["var"].isin(exclude_vars)].copy()

	gen_rows = []
	for t in range(numTotalPeriods):
		year = t + 1
		node_for_year = nodes[t // numSubperiods]
		for s in range(numSubterms):
			e_val = sp_data["Electricity Generation"][t][s]
			h_val = sp_data["Heat Generation"][t][s]
			ed_val = electricity_demand[t][s]
			hd_val = heat_demand[t][s]
			gen_rows.append({"var": "electricitygeneration", "node": node_for_year, "year": year, "subterm": s + 1, "value": e_val})
			gen_rows.append({"var": "heatgeneration", "node": node_for_year, "year": year, "subterm": s + 1, "value": h_val})
			gen_rows.append({"var": "electricitydemand", "node": node_for_year, "year": year, "subterm": s + 1, "value": ed_val})
			gen_rows.append({"var": "heatdemand", "node": node_for_year, "year": year, "subterm": s + 1, "value": hd_val})

	df_gen = pd.DataFrame(gen_rows)
	df_filtered = pd.concat([df_filtered, df_gen], ignore_index=True)

	df_filtered["x_index"] = (df_filtered["year"] - 1) * numSubterms + df_filtered["subterm"]

	fig = px.line(
		data_frame=df_filtered,
		x="x_index",
		y="value",
		color="var",
		hover_data=["node", "year"],
		title=f"Nodes: {nodes}"
	)
	fig.update_layout(xaxis_title="subterm", yaxis_title="value")
	fig.show()

if __name__ == "__main__":
	main()