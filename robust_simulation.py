import os
import re
import csv
import copy
import time
import numpy as np
import concurrent.futures
from fetch_data import fetch_data
from collections import defaultdict
from gurobipy import GRB, Model, quicksum
from scenario_tree import generate_scenario_tree, extract_stage_node_ranges, extract_scenario_paths_and_probabilities

_worker_state = None

def _init_replication_worker(scenario_tree_copy, stage_node_ranges, input_data, numStages, numSubperiods, numSubterms, subterm_interval_length, perturbed_subterm_interval_length):
    global _worker_state

    _worker_state = {
        'scenario_tree': scenario_tree_copy,
        'stage_node_ranges': stage_node_ranges,
        'input_data': input_data,
        'numStages': numStages,
        'numSubperiods': numSubperiods,
        'numSubterms': numSubterms,
        'total_periods': numStages * numSubperiods,
        'subterm_interval_length': subterm_interval_length,
        'perturbed_subterm_interval_length': perturbed_subterm_interval_length,
        'nominal_e_demand': input_data['electricity_demand'],
        'nominal_h_demand': input_data['heat_demand'],
        'e_demand_std': input_data['electricity_demand_std'],
        'h_demand_std': input_data['heat_demand_std'],
        'random_data': ['electricity_demand', 'heat_demand', 'solar', 'wind', 'parabolic_trough', 'heat_pump'],
    }

def run_single_replication(args):
    replication_index, seed = args
    ws = _worker_state

    rng = np.random.default_rng(seed)

    scenario_tree = ws['scenario_tree']
    stage_node_ranges = ws['stage_node_ranges']
    input_data = ws['input_data']
    numStages = ws['numStages']
    numSubperiods = ws['numSubperiods']
    numSubterms = ws['numSubterms']
    total_periods = ws['total_periods']
    subterm_interval_length = ws['subterm_interval_length']
    perturbed_subterm_interval_length = ws['perturbed_subterm_interval_length']
    nominal_e_demand = ws['nominal_e_demand']
    nominal_h_demand = ws['nominal_h_demand']
    e_demand_std = ws['e_demand_std']
    h_demand_std = ws['h_demand_std']
    random_data_sources = ws['random_data']

    replication_start_time = time.time()

    random_number_stream = {}
    for source in random_data_sources:
        random_number_stream[source] = {0: np.zeros(numSubterms)}
        for period in range(1, total_periods + 1):
            random_number_stream[source][period] = rng.standard_normal(numSubterms)
            #random_number_stream[source][period] = np.zeros(numSubterms)

    perturbed_e_demand = [None]
    perturbed_h_demand = [None]
    for period in range(1, total_periods + 1):
        perturbed_e_demand.append([max(0, nominal_e_demand[period][p] + random_number_stream['electricity_demand'][period][p] * e_demand_std[period][p]) for p in range(numSubterms)])
        perturbed_h_demand.append([max(0, nominal_h_demand[period][p] + random_number_stream['heat_demand'][period][p] * h_demand_std[period][p]) for p in range(numSubterms)])

    def get_perturbation_z(period, subterm_index):
        return {
            'solar': random_number_stream['solar'][period][subterm_index],
            'wind': random_number_stream['wind'][period][subterm_index],
            'parabolic_trough': random_number_stream['parabolic_trough'][period][subterm_index],
            'heat_pump': random_number_stream['heat_pump'][period][subterm_index],
        }

    rows = []
    prev_carry_values = {}

    for stage_no in range(1, numStages + 1):
        for solve_node_id in stage_node_ranges[stage_no]:
            solve_node = scenario_tree.nodes[solve_node_id]

            model = Model('Dispatch')
            model.setParam('OutputFlag', 1)
            model.setParam('Threads', 1)
            model.setParam('LogToConsole', 0)
            #model.setParam('LogFile', os.path.join(input_data['results_directory'], f'DispatchGurobiLog_rep{replication_index}.txt'))

            solve_node.InitializeModel(model, subterm_interval_length)

            first_period = (stage_no - 1) * numSubperiods + 1
            last_period = stage_no * numSubperiods

            for period in range(first_period, last_period + 1):
                if period == first_period:
                    if period > 1:
                        prev_node_id = solve_node.FindAncestorFromDiff(period - 1, period).id
                        if prev_node_id in prev_carry_values:
                            solve_node.e_Carrying[0].lb = prev_carry_values[prev_node_id]['e']
                            solve_node.e_Carrying[0].ub = prev_carry_values[prev_node_id]['e']
                            solve_node.h_Carrying[0].lb = prev_carry_values[prev_node_id]['h']
                            solve_node.h_Carrying[0].ub = prev_carry_values[prev_node_id]['h']

                    initial_subterms = list(range(1, subterm_interval_length + 1))
                    solve_node.InitializeConstraints(model, subterm_interval_length, period, initial_subterms, nominal_e_demand, nominal_h_demand, input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                    slot_mapped_periods = {s: period for s in range(1, subterm_interval_length+1)}

                    for s in range(1, perturbed_subterm_interval_length + 1):
                        p_z = get_perturbation_z(period, s - 1)
                        solve_node.UpdateDemandData(model, s, period, s, perturbed_e_demand, perturbed_h_demand, perturbation_z=p_z)

                else:
                    solve_node.e_Carrying[0].lb = solve_node.e_Carrying[1].X
                    solve_node.e_Carrying[0].ub = solve_node.e_Carrying[1].X
                    solve_node.h_Carrying[0].lb = solve_node.h_Carrying[1].X
                    solve_node.h_Carrying[0].ub = solve_node.h_Carrying[1].X

                    for s in range(1, subterm_interval_length+1):
                        solve_node.UpdateSubperiodData(model, s, period, input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                        if s <= perturbed_subterm_interval_length:
                            p_z = get_perturbation_z(period, s - 1)
                            solve_node.UpdateDemandData(model, s, period, s, perturbed_e_demand, perturbed_h_demand, perturbation_z=p_z)
                        else:
                            solve_node.UpdateDemandData(model, s, period, s, nominal_e_demand, nominal_h_demand)
                    slot_mapped_periods = {s: period for s in range(1, subterm_interval_length+1)}

                last_active_slot = subterm_interval_length

                for subterm_no in range(1, numSubterms + 1):
                    if subterm_no > 1:
                        solve_node.e_Carrying[0].lb = solve_node.e_Carrying[1].X
                        solve_node.e_Carrying[0].ub = solve_node.e_Carrying[1].X
                        solve_node.h_Carrying[0].lb = solve_node.h_Carrying[1].X
                        solve_node.h_Carrying[0].ub = solve_node.h_Carrying[1].X

                        for s in range(1, subterm_interval_length+1):
                            mapped_subterm_raw = subterm_no + s - 1
                            mapped_period = solve_node._get_mapped_period(period, mapped_subterm_raw)
                            mapped_subterm = solve_node._get_mapped_subterm(mapped_subterm_raw)

                            mapped_stage = ((mapped_period - 1) // numSubperiods) + 1
                            if mapped_period > total_periods or (mapped_period > period and mapped_stage != stage_no):
                                solve_node.DeactivateSlot(model, s)
                                if s == last_active_slot:
                                    last_active_slot -= 1
                            else:
                                if mapped_period != slot_mapped_periods[s]:
                                    solve_node.UpdateSubperiodData(model, s, mapped_period, input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                                    slot_mapped_periods[s] = mapped_period

                                if s <= perturbed_subterm_interval_length:
                                    p_z = get_perturbation_z(mapped_period, mapped_subterm - 1)
                                    solve_node.UpdateDemandData(model, s, mapped_period, mapped_subterm, perturbed_e_demand, perturbed_h_demand, perturbation_z=p_z)
                                else:
                                    solve_node.UpdateDemandData(model, s, mapped_period, mapped_subterm, nominal_e_demand, nominal_h_demand)

                    if solve_node.e_Carrying[last_active_slot].Obj != -1e-4:
                        solve_node.e_Carrying[last_active_slot].Obj = -1e-4
                        solve_node.h_Carrying[last_active_slot].Obj = -1e-4

                    model.update()
                    model.optimize()

                    rows.append((replication_index, solve_node_id, period, subterm_no, solve_node.e_Purchase[1].X, solve_node.h_Purchase[1].X))

                prev_carry_values[solve_node_id] = {'e': solve_node.e_Carrying[1].X, 'h': solve_node.h_Carrying[1].X}

            del model

    elapsed = time.time() - replication_start_time
    return replication_index, rows, elapsed

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

    def ComputeElectricityGeneration(self, t_, p, perturbation_z=None):
        total = 0.0
        for i, tech in enumerate(self.electricitygenerationtechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_+1):
                    if t <= t_ < t + self.FindAncestorFromDiff(t, t_).electricitygenerationtechNodeList[i].lifetime[v]:
                        ancestor_tech = self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i]
                        gen = ancestor_tech.periodic_electricity[v][p]
                        if perturbation_z is not None and ancestor_tech.periodic_electricity_std is not None:
                            z = perturbation_z.get(tech.tree.type, 0)
                            gen = max(0, gen + z * ancestor_tech.periodic_electricity_std[v][p])
                        total += gen * (1 - (ancestor_tech.degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
        return total

    def ComputeHeatGeneration(self, t_, p, perturbation_z=None):
        total = 0.0
        for i, tech in enumerate(self.heatgenerationtechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_+1):
                    if t <= t_ < t + self.FindAncestorFromDiff(t, t_).heatgenerationtechNodeList[i].lifetime[v]:
                        ancestor_tech = self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i]
                        gen = ancestor_tech.periodic_heat[v][p]
                        if perturbation_z is not None and ancestor_tech.periodic_heat_std is not None:
                            z = perturbation_z.get(tech.tree.type, 0)
                            gen = max(0, gen + z * ancestor_tech.periodic_heat_std[v][p])
                        total += gen * (1 - (ancestor_tech.degradation_rate[v] * (t_ - t))) * self.GetPlusValue(tech.tree.type, v, t, t_)
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

    def InitializeModel(self, model, subterm_interval_length):
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

        for s in range(1, subterm_interval_length+1):
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
        for s in range(1, subterm_interval_length+1):
            for (i, tech, v, t) in all_ht_keys:
                self.y_Transfer[s, tech.tree.type, v, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"transferredheat_{self.id}_{s}_{tech.tree.type}_{v}_{t}")

        model.update()

    def InitializeConstraints(self, model, subterm_interval_length, subperiod, subterms_in_window, electricity_demand, heat_demand, electricity_purchasing_cost, heat_purchasing_cost, discount_factor, perturbation_z=None):
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

        for s in range(1, subterm_interval_length+1):
            p = subterms_in_window[s-1] - 1

            if self.id != 0:
                self.e_Purchase[s].Obj = self.probability * electricity_purchasing_cost[subperiod] * (discount_factor ** subperiod)
                self.h_Purchase[s].Obj = self.probability * heat_purchasing_cost[subperiod] * (discount_factor ** subperiod)

            gen_e = self.ComputeElectricityGeneration(subperiod, p, perturbation_z=perturbation_z)
            self.demand_e_gen_constrs[s] = model.addConstr(self.e_Purchase[s] - self.e_Charging[s] + self.e_Discharging[s] - self.e_Satisfied[s] >= -gen_e, name=f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{s}')

            valid_ht_keys = self.GetValidHeatTransferKeys(subperiod)

            cop_coeffs = {}
            for (i, tech, v, t) in valid_ht_keys:
                ancestor_tech = self.FindAncestorFromDiff(t, subperiod).heattransfertechNodeList[i]
                cop = ancestor_tech.periodic_heat_transfer_cop[v][p]
                if perturbation_z is not None and ancestor_tech.periodic_heat_transfer_cop_std is not None:
                    z = perturbation_z.get(tech.tree.type, 0)
                    cop = max(1e-6, cop + z * ancestor_tech.periodic_heat_transfer_cop_std[v][p])
                cop_coeffs[(i, tech, v, t)] = cop

            self.demand_e_constrs[s] = model.addConstr(quicksum((-1 / cop_coeffs[(i, tech, v, t)]) * self.y_Transfer[s, tech.tree.type, v, t] for (i, tech, v, t) in valid_ht_keys) + self.e_Satisfied[s] >= electricity_demand[subperiod][p], name=f'N{self.id}_Demand_Electricity_{s}')

            gen_h = self.ComputeHeatGeneration(subperiod, p, perturbation_z=perturbation_z)
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

    def UpdateDemandData(self, model, s, mapped_subperiod, mapped_subterm, electricity_demand, heat_demand, perturbation_z=None):
        p = mapped_subterm - 1

        gen_e = self.ComputeElectricityGeneration(mapped_subperiod, p, perturbation_z=perturbation_z)
        self.demand_e_gen_constrs[s].RHS = -gen_e
        self.demand_e_constrs[s].RHS = electricity_demand[mapped_subperiod][p]

        gen_h = self.ComputeHeatGeneration(mapped_subperiod, p, perturbation_z=perturbation_z)
        self.demand_h_gen_constrs[s].RHS = -gen_h
        self.demand_h_constrs[s].RHS = heat_demand[mapped_subperiod][p]

        all_ht_keys = self.GetAllHeatTransferKeys()
        valid_ht_keys = set()
        for (i, tech, v, t) in self.GetValidHeatTransferKeys(mapped_subperiod):
            valid_ht_keys.add((i, tech.tree.type, v, t))

        for (i, tech, v, t) in all_ht_keys:
            if (i, tech.tree.type, v, t) in valid_ht_keys:
                ancestor_tech = self.FindAncestorFromDiff(t, mapped_subperiod).heattransfertechNodeList[i]
                cop = ancestor_tech.periodic_heat_transfer_cop[v][p]
                if perturbation_z is not None and ancestor_tech.periodic_heat_transfer_cop_std is not None:
                    z = perturbation_z.get(tech.tree.type, 0)
                    cop = max(1e-6, cop + z * ancestor_tech.periodic_heat_transfer_cop_std[v][p])
                model.chgCoeff(self.demand_e_constrs[s], self.y_Transfer[s, tech.tree.type, v, t], (-1.0 / cop))
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

def parse(s: str):
    m = re.match(r'([a-zA-Z]+)_(\d+)\[([^\]]+)\]', s)

    dv_name = m.group(1).strip()
    node_id = int(m.group(2).strip())
    indices = [p.strip() for p in m.group(3).split(',')]

    return dv_name, node_id, indices

if __name__ == '__main__':
    execution_start_time = time.time()

    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2
    tolerance = 0.01

    subterm_interval_length = 12
    perturbed_subterm_interval_length = 3
    numReplications = 100

    num_workers = min(os.cpu_count(), numReplications)

    input_data = fetch_data(numStages, numSubperiods, numSubterms)

    results_sol_path = os.path.join(input_data['results_directory'], 'Results.sol')

    scenario_tree, initial_tech = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, dispatch_flag=True, solar_periodic_generation_std=input_data['solar_periodic_generation_std'], wind_periodic_generation_std=input_data['wind_periodic_generation_std'], parabolic_trough_periodic_generation_std=input_data['parabolic_trough_periodic_generation_std'], heat_pump_cop_std=input_data['heat_pump_cop_std'])

    stage_node_ranges = extract_stage_node_ranges(scenario_tree)
    scenario_paths, scenario_path_probabilities = extract_scenario_paths_and_probabilities(scenario_tree)

    plus_vars = {}
    opt_results = {}
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
                elif var_name.startswith("electricitypurchase_") or var_name.startswith("heatpurchase_"):
                    dv_name, node_id, indices = parse(var_name)
                    if node_id == 0:
                        continue
                    key = (node_id, int(indices[0]), int(indices[1]))
                    if key not in opt_results:
                        opt_results[key] = {'e': 0.0, 'h': 0.0}
                    if "electricitypurchase" in dv_name:
                        opt_results[key]['e'] = float(parts[1])
                    elif "heatpurchase" in dv_name:
                        opt_results[key]['h'] = float(parts[1])

    scenario_tree.plus_vars = plus_vars

    master_seed_seq = np.random.SeedSequence(42)
    child_seeds = master_seed_seq.spawn(numReplications)

    all_results = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_replication_worker,
        initargs=(copy.deepcopy(scenario_tree), stage_node_ranges, input_data, numStages, numSubperiods, numSubterms, subterm_interval_length, perturbed_subterm_interval_length)
    ) as executor:
        futures = {executor.submit(run_single_replication, (r, child_seeds[r-1])): r for r in range(1, numReplications + 1)}

        for future in concurrent.futures.as_completed(futures):
            replication_index, rows, elapsed = future.result()
            all_results[replication_index] = rows
            print(f"Replication {replication_index} completed in {elapsed:.2f} seconds")

    opt_rows = []
    for (node_id, subperiod, subterm), values in opt_results.items():
        opt_rows.append(('optimization', node_id, subperiod, subterm, values['e'], values['h']))

    all_results['optimization'] = opt_rows

    csv_path = os.path.join(input_data['results_directory'], 'robust_simulation_results.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['replication', 'node_id', 'subperiod', 'subterm', 'electricity_purchase', 'heat_purchase'])
        
        int_keys = sorted([k for k in all_results.keys() if isinstance(k, int)])
        csv_writer.writerows(all_results['optimization'])
        for r in int_keys:
            csv_writer.writerows(all_results[r])

    annual = defaultdict(lambda: {'electricity': 0.0, 'heat': 0.0})
    for key, rows in all_results.items():
        for rep, node_id, subperiod, _subterm, e_purch, h_purch in rows:
            annual[(rep, node_id, subperiod)]['electricity'] += e_purch
            annual[(rep, node_id, subperiod)]['heat'] += h_purch

    opt_agg_keys = sorted([k for k in annual if k[0] == 'optimization'], key=lambda x: (x[1], x[2]))
    rep_agg_keys = sorted([k for k in annual if isinstance(k[0], int)], key=lambda x: (x[0], x[1], x[2]))

    annual_csv_path = os.path.join(input_data['results_directory'], 'annual_purchases.csv')
    with open(annual_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['replication', 'node_id', 'subperiod', 'annual_electricity_purchase', 'annual_heat_purchase'])
        for k in opt_agg_keys + rep_agg_keys:
            csv_writer.writerow([k[0], k[1], k[2], annual[k]['electricity'], annual[k]['heat']])

    print(f"Total Execution Time: {time.time() - execution_start_time:.2f} seconds")