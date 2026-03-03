import os
import csv
import time
from fetch_data import fetch_data
from gurobipy import GRB, Model, quicksum
from scenario_tree import generate_scenario_tree, extract_stage_node_ranges, extract_scenario_paths_and_probabilities

class ScenarioNodeDispatch:
    def __init__(self, id_In, parent_In, probability_In, tree_In, techNodeList_In):
        self.id = id_In
        self.parent = parent_In
        self.tree = tree_In

        if self.parent is None:
            self.stage = 0
            self.numSubperiods = 1
            self.stageSubperiods = [0]
            self.allSubperiods = [0]
        else:
            self.stage = self.parent.stage + 1
            self.numSubperiods = self.tree.numSubperiods
            self.stageSubperiods = [1 + (self.stage-1) * self.numSubperiods + t for t in range(self.numSubperiods)]
            self.allSubperiods = [0] + [1 + (s-1) * self.numSubperiods + t for s in range(1,self.stage+1) for t in range(self.numSubperiods)]

        self.numSubterms = self.tree.numSubterms
        self.stageSubterms = [p for p in range(1, self.numSubterms+1)]
        self.probability = probability_In
        self.techNodeList = techNodeList_In
        self.electricitygenerationtechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'electricity generation']
        self.heatgenerationtechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'heat generation']
        self.electricitystoragetechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'electricity storage']
        self.heatstoragetechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'heat storage']
        self.heattransfertechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'heat transfer']

        self.tech_types = [tech.tree.type for tech in self.techNodeList]
        self.tree.nodes.append(self)
        self.children = []

    def AddChild(self, techNodeList):
        prob = 1
        for techNode in techNodeList:
            prob *= techNode.probability
        child = ScenarioNodeDispatch(len(self.tree.nodes), self, prob, self.tree, techNodeList)
        self.children.append(child)

    def FindAncestorFromDiff(self, t, t_):
        ancestor = self
        amount_subperiods = len(ancestor.stageSubperiods) 
        node1_stage = (t-1) // amount_subperiods
        node2_stage = (t_-1) // amount_subperiods
        ancestor_degree = node2_stage - node1_stage
        for _ in range(ancestor_degree):
            ancestor = ancestor.parent
        return ancestor

    def GetPlusValue(self, tech_type, version, t, t_):
        ancestor = self.FindAncestorFromDiff(t, t_)
        key = f"plus_{ancestor.id}[{tech_type},{version},{t}]"
        if key in self.tree.plus_vars:
            return self.tree.plus_vars[key]
        key_with_spaces = f"plus_{ancestor.id}[{tech_type}, {version}, {t}]"
        return self.tree.plus_vars.get(key_with_spaces, 0.0)

    def ComputeElectricityGeneration(self, t_, p):
        total = 0.0
        for i, tech in enumerate(self.electricitygenerationtechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_+1):
                    if t <= t_ < t + self.FindAncestorFromDiff(t, t_).electricitygenerationtechNodeList[i].lifetime[v]:
                        total += self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p] * (1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
        return total

    def ComputeHeatGeneration(self, t_, p):
        total = 0.0
        for i, tech in enumerate(self.heatgenerationtechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_+1):
                    if t <= t_ < t + self.FindAncestorFromDiff(t, t_).heatgenerationtechNodeList[i].lifetime[v]:
                        total += self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p] * (1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
        return total

    def ComputeElectricityStorageCapacity(self, t_):
        total = 0.0
        for i, tech in enumerate(self.electricitystoragetechNodeList):
            for v in range(tech.NumVersion):
                for t in self.allSubperiods:
                    if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]:
                        total += self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v] * (1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
        return total

    def ComputeHeatStorageCapacity(self, t_):
        total = 0.0
        for i, tech in enumerate(self.heatstoragetechNodeList):
            for v in range(tech.NumVersion):
                for t in self.allSubperiods:
                    if t <= t_ < t + self.FindAncestorFromDiff(t, t_).heatstoragetechNodeList[i].lifetime[v]:
                        total += self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v] * (1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
        return total

    def ComputeHeatTransferCapacity(self, tech, v, t, t_):
        cap = self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[next(idx for idx, tt in enumerate(self.heattransfertechNodeList) if tt is tech)].heat_transfer_capacity[v]
        return cap * self.GetPlusValue(tech.tree.type, v, t, t_)

    def GetValidHeatTransferKeys(self, t_):
        keys = []
        for i, tech in enumerate(self.heattransfertechNodeList):
            for v in range(tech.NumVersion):
                for t in self.allSubperiods:
                    if t <= t_ < t + tech.lifetime[v]:
                        keys.append((i, tech, v, t))
        return keys

    def GetAllHeatTransferKeys(self):
        keys = []
        for i, tech in enumerate(self.heattransfertechNodeList):
            for v in range(tech.NumVersion):
                for t in self.allSubperiods:
                    keys.append((i, tech, v, t))
        return keys

    def InitializeModel(self, model, L):
        self.e_Purchase = {}
        self.h_Purchase = {}
        self.e_Charging = {}
        self.h_Charging = {}
        self.e_Discharging = {}
        self.h_Discharging = {}
        self.e_Satisfied = {}
        self.h_Satisfied = {}
        self.y_Transfer = {}
        self.e_Carrying = {}
        self.h_Carrying = {}
        self.e_Carrying[0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"electricitycarry_{self.id}_0")
        self.h_Carrying[0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"heatcarry_{self.id}_0")

        for s in range(1, L+1):
            self.e_Purchase[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"electricitypurchase_{self.id}_{s}")
            self.h_Purchase[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"heatpurchase_{self.id}_{s}")
            self.e_Carrying[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"electricitycarry_{self.id}_{s}")
            self.h_Carrying[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"heatcarry_{self.id}_{s}")
            self.e_Charging[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"electricitycharge_{self.id}_{s}")
            self.h_Charging[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"heatcharge_{self.id}_{s}")
            self.e_Discharging[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"electricitydischarge_{self.id}_{s}")
            self.h_Discharging[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"heatdischarge_{self.id}_{s}")
            self.e_Satisfied[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"electricityused_{self.id}_{s}")
            self.h_Satisfied[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"heatused_{self.id}_{s}")

        all_ht_keys = self.GetAllHeatTransferKeys()
        for s in range(1, L+1):
            for (i, tech, v, t) in all_ht_keys:
                self.y_Transfer[s, tech.tree.type, v, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"transferredheat_{self.id}_{s}_{tech.tree.type}_{v}_{t}")

        model.update()

    def InitializeConstraints(self, model, L, subperiod, subterms_in_window, electricity_demand, heat_demand, electricity_purchasing_cost, heat_purchasing_cost, discount_factor):
        all_ht_keys = self.GetAllHeatTransferKeys()

        self.demand_e_gen_constrs = {}
        self.demand_e_constrs = {}
        self.demand_h_gen_constrs = {}
        self.demand_h_constrs = {}
        self.inv_bal_e_constrs = {}
        self.inv_bal_h_constrs = {}
        self.storage_cap_e_constrs = {}
        self.storage_cap_h_constrs = {}
        self.heat_transfer_cap_constrs = {}

        for s in range(1, L+1):
            p = subterms_in_window[s-1] - 1

            if self.id != 0:
                self.e_Purchase[s].Obj = self.probability * electricity_purchasing_cost[subperiod] * (discount_factor ** subperiod)
                self.h_Purchase[s].Obj = self.probability * heat_purchasing_cost[subperiod] * (discount_factor ** subperiod)

            gen_e = self.ComputeElectricityGeneration(subperiod, p)
            self.demand_e_gen_constrs[s] = model.addConstr(self.e_Purchase[s] - self.e_Charging[s] + self.e_Discharging[s] - self.e_Satisfied[s] >= -gen_e, name=f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{s}')

            valid_ht_keys = self.GetValidHeatTransferKeys(subperiod)
            self.demand_e_constrs[s] = model.addConstr(quicksum((-1 / self.FindAncestorFromDiff(t,subperiod).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p]) * self.y_Transfer[s, tech.tree.type, v, t] for (i, tech, v, t) in valid_ht_keys) + self.e_Satisfied[s] >= electricity_demand[subperiod][p], name=f'N{self.id}_Demand_Electricity_{s}')

            gen_h = self.ComputeHeatGeneration(subperiod, p)
            self.demand_h_gen_constrs[s] = model.addConstr(self.h_Purchase[s] - self.h_Charging[s] + self.h_Discharging[s] - self.h_Satisfied[s] >= -gen_h, name=f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{s}')

            self.demand_h_constrs[s] = model.addConstr(quicksum((1 - self.FindAncestorFromDiff(t, subperiod).heattransfertechNodeList[i].degradation_rate[v] * (subperiod - t)) * self.y_Transfer[s, tech.tree.type, v, t] for (i, tech, v, t) in valid_ht_keys) + self.h_Satisfied[s] >= heat_demand[subperiod][p], name=f'N{self.id}_Demand_Heat_{s}')

            self.inv_bal_e_constrs[s] = model.addConstr(self.e_Carrying[s] - self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * self.e_Carrying[s-1] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[s] + (1/self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[s] == 0, name=f'N{self.id}_ElectricityInventoryBalance_{s}')
            self.inv_bal_h_constrs[s] = model.addConstr(self.h_Carrying[s] - self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * self.h_Carrying[s-1] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[s] + (1/self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[s] == 0, name=f'N{self.id}_HeatInventoryBalance_{s}')

            electricity_storage_capacity = self.ComputeElectricityStorageCapacity(subperiod)
            heat_storage_capacity = self.ComputeHeatStorageCapacity(subperiod)
            self.storage_cap_e_constrs[s] = model.addConstr(-self.e_Carrying[s] >= -electricity_storage_capacity, name=f'N{self.id}_ElectricityStorageCapacity_{s}')
            self.storage_cap_h_constrs[s] = model.addConstr(-self.h_Carrying[s] >= -heat_storage_capacity, name=f'N{self.id}_HeatStorageCapacity_{s}')

            for (i, tech, v, t) in all_ht_keys:
                cap_val = self.ComputeHeatTransferCapacity(tech, v, t, subperiod) if t <= subperiod < t + tech.lifetime[v] else 0.0
                self.heat_transfer_cap_constrs[s, tech.tree.type, v, t] = model.addConstr(-self.y_Transfer[s, tech.tree.type, v, t] >= -cap_val, name=f'N{self.id}_Heat_Transfer_Capacity_{s}_{tech.tree.type}_{v}_{t}')

        model.update()

    def DeactivateSlot(self, model, s):
        all_vars = [self.e_Purchase[s], self.h_Purchase[s], self.e_Carrying[s], self.h_Carrying[s], self.e_Charging[s], self.h_Charging[s], self.e_Discharging[s], self.h_Discharging[s], self.e_Satisfied[s], self.h_Satisfied[s]]
        for (i, tech, v, t) in self.GetAllHeatTransferKeys():
            all_vars.append(self.y_Transfer[s, tech.tree.type, v, t])
        for var in all_vars:
            var.lb = 0
            var.ub = 0
        self.e_Purchase[s].Obj = 0
        self.h_Purchase[s].Obj = 0
        model.chgCoeff(self.inv_bal_e_constrs[s], self.e_Carrying[s-1], 0.0)
        model.chgCoeff(self.inv_bal_h_constrs[s], self.h_Carrying[s-1], 0.0)
        self.demand_e_constrs[s].RHS = 0
        self.demand_h_constrs[s].RHS = 0
        self.demand_e_gen_constrs[s].RHS = 0
        self.demand_h_gen_constrs[s].RHS = 0

    def UpdateSubperiodData(self, model, s, mapped_subperiod, electricity_purchasing_cost, heat_purchasing_cost, discount_factor):
        if self.id != 0:
            self.e_Purchase[s].Obj = self.probability * electricity_purchasing_cost[mapped_subperiod] * (discount_factor ** mapped_subperiod)
            self.h_Purchase[s].Obj = self.probability * heat_purchasing_cost[mapped_subperiod] * (discount_factor ** mapped_subperiod)

        all_ht_keys = self.GetAllHeatTransferKeys()
        valid_ht_keys = set()
        for (i, tech, v, t) in self.GetValidHeatTransferKeys(mapped_subperiod):
            valid_ht_keys.add((i, tech.tree.type, v, t))

        for (i, tech, v, t) in all_ht_keys:
            if (i, tech.tree.type, v, t) in valid_ht_keys:
                model.chgCoeff(self.demand_h_constrs[s], self.y_Transfer[s, tech.tree.type, v, t], 1.0 - (self.FindAncestorFromDiff(t, mapped_subperiod).heattransfertechNodeList[i].degradation_rate[v] * (mapped_subperiod - t)))
            else:
                model.chgCoeff(self.demand_h_constrs[s], self.y_Transfer[s, tech.tree.type, v, t], 0.0)

        self.storage_cap_e_constrs[s].RHS = -self.ComputeElectricityStorageCapacity(mapped_subperiod)
        self.storage_cap_h_constrs[s].RHS = -self.ComputeHeatStorageCapacity(mapped_subperiod)

        for (i, tech, v, t) in all_ht_keys:
            cap_val = self.ComputeHeatTransferCapacity(tech, v, t, mapped_subperiod) if t <= mapped_subperiod < t + tech.lifetime[v] else 0.0
            self.heat_transfer_cap_constrs[s, tech.tree.type, v, t].RHS = -cap_val

    def UpdateDemandData(self, model, s, mapped_subperiod, mapped_subterm, electricity_demand, heat_demand):
        p = mapped_subterm - 1

        gen_e = self.ComputeElectricityGeneration(mapped_subperiod, p)
        self.demand_e_gen_constrs[s].RHS = -gen_e
        self.demand_e_constrs[s].RHS = electricity_demand[mapped_subperiod][p]

        gen_h = self.ComputeHeatGeneration(mapped_subperiod, p)
        self.demand_h_gen_constrs[s].RHS = -gen_h
        self.demand_h_constrs[s].RHS = heat_demand[mapped_subperiod][p]

        all_ht_keys = self.GetAllHeatTransferKeys()
        valid_ht_keys = set()
        for (i, tech, v, t) in self.GetValidHeatTransferKeys(mapped_subperiod):
            valid_ht_keys.add((i, tech.tree.type, v, t))

        for (i, tech, v, t) in all_ht_keys:
            if (i, tech.tree.type, v, t) in valid_ht_keys:
                model.chgCoeff(self.demand_e_constrs[s], self.y_Transfer[s, tech.tree.type, v, t], (-1.0 / self.FindAncestorFromDiff(t, mapped_subperiod).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p]))
            else:
                model.chgCoeff(self.demand_e_constrs[s], self.y_Transfer[s, tech.tree.type, v, t], 0.0)

    def _get_mapped_period(self, base_period, base_subterm):
        """Get the period that an actual subterm belongs to."""
        if base_subterm <= self.numSubterms:
            return base_period
        else:
            return base_period + 1

    def _get_mapped_subterm(self, base_subterm):
        """Get the subterm number within its period."""
        if base_subterm <= self.numSubterms:
            return base_subterm
        else:
            return base_subterm - self.numSubterms

if __name__ == '__main__':
    execution_start_time = time.time()

    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2
    tolerance = 0.01

    subterm_interval_length = 12

    input_data = fetch_data(numStages, numSubperiods, numSubterms)

    results_sol_path = os.path.join(input_data['results_directory'], 'Results.sol')

    scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, dispatch_flag=True)

    stage_node_ranges = extract_stage_node_ranges(scenario_tree)
    scenario_paths, scenario_path_probabilities = extract_scenario_paths_and_probabilities(scenario_tree)

    total_periods = numStages * numSubperiods

    csv_path = os.path.join(input_data['results_directory'], 'dispatch_results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['node_id', 'subperiod', 'subterm', 'electricity_purchase', 'heat_purchase'])

    plus_vars = {}
    with open(results_sol_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                var_name = parts[0]
                if "plus" in var_name:
                    plus_vars[var_name] = float(parts[1])

    scenario_tree.plus_vars = plus_vars

    for node in scenario_tree.nodes:
        node.v_Plus = {}
    for key, val in plus_vars.items():
        if "plus_" in key:
            inner = key.split('[')[1].rstrip(']')
            parts_inner = [p.strip() for p in inner.split(',')]
            scenario_tree.nodes[int(key.split('_')[1].split('[')[0])].v_Plus[parts_inner[0], int(parts_inner[1]), int(parts_inner[2])] = val

    prev_carry_values = {}
    L = subterm_interval_length

    for stage_no in range(1, numStages + 1):
        for solve_node_id in stage_node_ranges[stage_no]:
            solve_node = scenario_tree.nodes[solve_node_id]

            model = Model('Dispatch')
            model.setParam('OutputFlag', True)
            model.setParam('Threads', 1)
            model.setParam('LogToConsole', 0)

            solve_node.InitializeModel(model, L)

            first_period = (stage_no - 1) * numSubperiods + 1
            last_period = stage_no * numSubperiods

            for period in range(first_period, last_period + 1):
                subperiod_start_time = time.time()
                subperiod_optimization_time = 0

                if period == first_period:
                    if period > 1:
                        prev_node_id = solve_node.FindAncestorFromDiff(period - 1, period).id
                        if prev_node_id in prev_carry_values:
                            solve_node.e_Carrying[0].lb = prev_carry_values[prev_node_id]['e']
                            solve_node.e_Carrying[0].ub = prev_carry_values[prev_node_id]['e']
                            solve_node.h_Carrying[0].lb = prev_carry_values[prev_node_id]['h']
                            solve_node.h_Carrying[0].ub = prev_carry_values[prev_node_id]['h']

                    initial_subterms = list(range(1, L + 1))
                    solve_node.InitializeConstraints(model, L, period, initial_subterms, input_data['electricity_demand'], input_data['heat_demand'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                    slot_mapped_periods = {s: period for s in range(1, L+1)}
                else:
                    solve_node.e_Carrying[0].lb = solve_node.e_Carrying[1].X
                    solve_node.e_Carrying[0].ub = solve_node.e_Carrying[1].X
                    solve_node.h_Carrying[0].lb = solve_node.h_Carrying[1].X
                    solve_node.h_Carrying[0].ub = solve_node.h_Carrying[1].X

                    for s in range(1, L+1):
                        solve_node.UpdateSubperiodData(model, s, period, input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                        solve_node.UpdateDemandData(model, s, period, s, input_data['electricity_demand'], input_data['heat_demand'])
                    slot_mapped_periods = {s: period for s in range(1, L+1)}

                for subterm_no in range(1, numSubterms + 1):
                    if subterm_no > 1:
                        solve_node.e_Carrying[0].lb = solve_node.e_Carrying[1].X
                        solve_node.e_Carrying[0].ub = solve_node.e_Carrying[1].X
                        solve_node.h_Carrying[0].lb = solve_node.h_Carrying[1].X
                        solve_node.h_Carrying[0].ub = solve_node.h_Carrying[1].X

                        for s in range(1, L+1):
                            mapped_subterm = subterm_no + s - 1
                            mapped_period = solve_node._get_mapped_period(period, mapped_subterm)
                            mapped_subterm = solve_node._get_mapped_subterm(mapped_subterm)

                            mapped_stage = ((mapped_period - 1) // numSubperiods) + 1
                            if mapped_period > total_periods or (mapped_period > period and mapped_stage != stage_no):
                                solve_node.DeactivateSlot(model, s)
                            else:
                                if mapped_period != slot_mapped_periods[s]:
                                    solve_node.UpdateSubperiodData(model, s, mapped_period, input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                                    slot_mapped_periods[s] = mapped_period
                                solve_node.UpdateDemandData(model, s, mapped_period, mapped_subterm, input_data['electricity_demand'], input_data['heat_demand'])

                    model.update()
                    model.optimize()
                    subperiod_optimization_time += model.Runtime

                    csv_writer.writerow([solve_node_id, period, subterm_no, solve_node.e_Purchase[1].X, solve_node.h_Purchase[1].X])

                prev_carry_values[solve_node_id] = {'e': solve_node.e_Carrying[1].X, 'h': solve_node.h_Carrying[1].X}

                print(f"Subperiod {period} completed in {time.time() - subperiod_start_time:.2f} seconds")
                print(f"Subperiod {period} optimization time {subperiod_optimization_time:.2f} seconds")

            del model

    csv_file.close()
    print(f"Total Execution Time: {time.time() - execution_start_time:.2f} seconds")