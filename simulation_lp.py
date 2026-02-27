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

    def AddInvestmentVariables(self, carry_nodes, period, subterm, model):
        continuous_tech_nodes = [x for x in self.techNodeList if x not in self.electricitygenerationtechNodeList]
        self.v_Plus = {}
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in continuous_tech_nodes for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.electricitygenerationtechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        self.e_Purchase = {}
        self.h_Purchase = {}
        self.e_Carrying = {}
        self.h_Carrying = {}
        self.e_Charging = {}
        self.h_Charging = {}
        self.e_Discharging = {}
        self.h_Discharging = {}
        self.e_Satisfied = {}
        self.h_Satisfied = {}
        self.y_Transfer = {}
        if self.id in carry_nodes:
            self.e_Carrying.update(model.addVars([(period - 1, subterm)], vtype=GRB.CONTINUOUS, lb=0, ub=0, name="electricitycarry_"+str(self.id))) # inventory carriage amount
            self.h_Carrying.update(model.addVars([(period - 1, subterm)], vtype=GRB.CONTINUOUS, lb=0, ub=0, name="heatcarry_"+str(self.id))) # inventory carriage amount

    def AddVariables(self, model, subperiod, subterm):
        self.e_Purchase.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="electricitypurchase_"+str(self.id))) # electricity purchase amount
        self.h_Purchase.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="heatpurchase_"+str(self.id))) # heat purchase amount
        self.e_Carrying.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="electricitycarry_"+str(self.id))) # inventory carriage amount
        self.h_Carrying.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="heatcarry_"+str(self.id))) # inventory carriage amount
        self.e_Charging.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="electricitycharge_"+str(self.id))) # inventory charge amount
        self.h_Charging.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="heatcharge_"+str(self.id))) # inventory charge amount
        self.e_Discharging.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="electricitydischarge_"+str(self.id))) # inventory discharge amount
        self.h_Discharging.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="heatdischarge_"+str(self.id))) # inventory discharge amount
        self.e_Satisfied.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="electricityused_"+str(self.id))) # electricity used from inventory and generation
        self.h_Satisfied.update(model.addVars([(subperiod, subterm)], vtype=GRB.CONTINUOUS, lb=0, name="heatused_"+str(self.id))) # heat used from inventory and generation
        self.y_Transfer.update(model.addVars([(subterm, tech.tree.type, v, t, subperiod) for tech in self.heattransfertechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= subperiod < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, lb=0, name="transferredheat_"+str(self.id))) # electricity transfered to heat

    def AddObjectiveCoefficients(self, electricity_purchasing_cost, heat_purchasing_cost, discount_factor, subperiod, subterm):
        t = subperiod
        p = subterm
        if self.id != 0:
            self.e_Purchase[t,p].Obj = self.probability * electricity_purchasing_cost[t] * (discount_factor**(t))
            self.h_Purchase[t,p].Obj = self.probability * heat_purchasing_cost[t] * (discount_factor**(t))

    def AddDemandConstraints(self, model, electricity_demand, heat_demand, subperiod, subterm):
        t_ = subperiod
        if self.id != 0:
            p = subterm-1
            periodic_demand = electricity_demand[t_][p]
            model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.electricitygenerationtechNodeList) for v in range(self.electricitygenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].lifetime[v]) + self.e_Purchase[t_, p+1] - self.e_Charging[t_, p+1] + self.e_Discharging[t_, p+1] >= self.e_Satisfied[t_, p+1], name = f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{t_}_{p}')
            model.addConstr(quicksum((-1/self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p])*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.e_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
            periodic_demand = heat_demand[t_][p]
            model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.heatgenerationtechNodeList) for v in range(self.heatgenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].lifetime[v]) + self.h_Purchase[t_, p+1] - self.h_Charging[t_, p+1] + self.h_Discharging[t_, p+1] >= self.h_Satisfied[t_, p+1], name = f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{t_}_{p}')
            model.addConstr(quicksum((1 - (self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].degradation_rate[v]*(t_ - t)))*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.h_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Heat_{t_}_{p}')

    def AddInventoryBalanceConstraints(self, model, subperiod, subterm):
        if self.id != 0:
            t_ = subperiod
            p = subterm
            if p == 1:
                model.addConstr(self.e_Carrying[t_,p] == self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * self.FindAncestorFromDiff(t_-1,t_).e_Carrying[t_-1, self.numSubterms] + self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] - (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p], name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                model.addConstr(self.h_Carrying[t_,p] == self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * self.FindAncestorFromDiff(t_-1,t_).h_Carrying[t_-1, self.numSubterms] + self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] - (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p], name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')
            else:
                model.addConstr(self.e_Carrying[t_,p] == self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * self.e_Carrying[t_,p-1] + self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] - (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p], name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                model.addConstr(self.h_Carrying[t_,p] == self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * self.h_Carrying[t_,p-1] + self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] - (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p], name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')

    def AddStorageCapacityConstraints(self, model, subperiod, subterm):
        t_ = subperiod
        p = subterm
        model.addConstr(self.e_Carrying[t_,p] <= quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]), name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{p}')
        model.addConstr(self.h_Carrying[t_,p] <= quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]), name = f'N{self.id}_HeatStorageCapacity_{t_}_{p}')

    def AddHeatTransferCapacityConstraints(self, model, subperiod, subterm):
        t_ = subperiod
        p = subterm
        for i, tech in enumerate(self.heattransfertechNodeList):
            for v in range(tech.NumVersion):
                for t in self.allSubperiods:
                    if t <= t_ < t + tech.lifetime[v]:
                        model.addConstr(self.y_Transfer[p,tech.tree.type,v,t,t_] <= self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].heat_transfer_capacity[v], name = f'N{self.id}_Heat_Transfer_Capacity_{tech.tree.type}_{v}_{t}_{t_}_{p}')

def Dispatch(carry_nodes, period, numSubterms, scenarioTree, results_directory, results_sol_path, model_name):
    model = Model(model_name)
    model.setParam('OutputFlag', True)

    for node in scenarioTree.nodes:
        node.AddInvestmentVariables(carry_nodes, period, numSubterms, model)

    model.update()

    if results_sol_path:
        plus_values = {}
        with open(results_sol_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    var_name = parts[0]
                    if "plus" in var_name:
                        var_value = float(parts[1])
                        plus_values[var_name] = var_value

        vars_to_fix = []
        values_to_fix = []

        for var in model.getVars():
            if var.varName in plus_values:
                vars_to_fix.append(var)
                values_to_fix.append(plus_values[var.varName])
        
        model.setAttr('LB', vars_to_fix, values_to_fix)
        model.setAttr('UB', vars_to_fix, values_to_fix)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)
    model.update()

    return model

def Interval(subperiod, subterm, current_nodes, scenarioTree, model, electricity_demand, heat_demand, electricity_purchasing_cost, heat_purchasing_cost, discount_factor):

    for node in scenarioTree.nodes:
        if node.id in current_nodes:
            node.AddVariables(model, subperiod, subterm)
            node.AddObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor, subperiod, subterm)
            node.AddDemandConstraints(model, electricity_demand, heat_demand, subperiod, subterm)
            node.AddInventoryBalanceConstraints(model, subperiod, subterm)
            node.AddStorageCapacityConstraints(model, subperiod, subterm)
            node.AddHeatTransferCapacityConstraints(model, subperiod, subterm)

    return model

if __name__ == '__main__':
    execution_start_time = time.time()

    numStages = 3
    numSubperiods = 5
    numSubterms = 4368
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
    csv_writer.writerow(['node_id', 'subperiod', 'subterm', 'electricity_purchase', 'heat_purchase', 'electricity_purchase_cost', 'heat_purchase_cost'])

    prev_carry_values = {}

    for period in range(1, total_periods + 1):
        subperiod_start_time = time.time()
        subperiod_optimization_time = 0
        stage_no = ((period - 1) // numSubperiods) + 1

        for solve_node_id in stage_node_ranges[stage_no]:
            solve_node = scenario_tree.nodes[solve_node_id]

            if period % numSubperiods == 1:
                carry_nodes = [solve_node.FindAncestorFromDiff(period - 1, period).id] if period > 1 else [0]
            else:
                carry_nodes = [solve_node_id]

            model = Dispatch(carry_nodes, period, numSubterms, scenario_tree, input_data['results_directory'], results_sol_path, model_name='Dispatch')

            if period > 1:
                prev_node = solve_node.FindAncestorFromDiff(period - 1, period)
                prev_node.e_Carrying[period - 1, numSubterms].lb = prev_carry_values[prev_node.id]['e']
                prev_node.e_Carrying[period - 1, numSubterms].ub = prev_carry_values[prev_node.id]['e']
                prev_node.h_Carrying[period - 1, numSubterms].lb = prev_carry_values[prev_node.id]['h']
                prev_node.h_Carrying[period - 1, numSubterms].ub = prev_carry_values[prev_node.id]['h']

            for i in range(1, subterm_interval_length):
                model = Interval(period, i, [solve_node_id], scenario_tree, model, input_data['electricity_demand'], input_data['heat_demand'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])

            for subterm_no in range(1, numSubterms + 1):
                next_to_add = subterm_no + subterm_interval_length - 1
                if next_to_add <= numSubterms:
                    model = Interval(period, next_to_add, [solve_node_id], scenario_tree, model, input_data['electricity_demand'], input_data['heat_demand'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                elif period < total_periods:
                    next_period = period + 1
                    next_stage_no = ((next_period - 1) // numSubperiods) + 1
                    new_st_no = next_to_add - numSubterms
                    if next_stage_no == stage_no:
                        model = Interval(next_period, new_st_no, [solve_node_id], scenario_tree, model, input_data['electricity_demand'], input_data['heat_demand'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])
                    else:
                        child_ids = [child.id for child in solve_node.children]
                        model = Interval(next_period, new_st_no, child_ids, scenario_tree, model, input_data['electricity_demand'], input_data['heat_demand'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], input_data['discount_factor'])

                model.update()
                model.optimize()
                subperiod_optimization_time += model.Runtime

                csv_writer.writerow([solve_node_id, period, subterm_no, solve_node.e_Purchase[period, subterm_no].X, solve_node.h_Purchase[period, subterm_no].X, solve_node.e_Purchase[period, subterm_no].X * solve_node.e_Purchase[period, subterm_no].Obj, solve_node.h_Purchase[period, subterm_no].X * solve_node.h_Purchase[period, subterm_no].Obj])

                vars_subterm = ([
                            f"electricitypurchase_{solve_node_id}[{period},{subterm_no}]",
                            f"heatpurchase_{solve_node_id}[{period},{subterm_no}]",
                            f"electricitycarry_{solve_node_id}[{period},{subterm_no}]",
                            f"heatcarry_{solve_node_id}[{period},{subterm_no}]",
                            f"electricitycharge_{solve_node_id}[{period},{subterm_no}]",
                            f"heatcharge_{solve_node_id}[{period},{subterm_no}]",
                            f"electricitydischarge_{solve_node_id}[{period},{subterm_no}]",
                            f"heatdischarge_{solve_node_id}[{period},{subterm_no}]",
                            f"electricityused_{solve_node_id}[{period},{subterm_no}]",
                            f"heatused_{solve_node_id}[{period},{subterm_no}]"]
                            + [f"transferredheat_{solve_node_id}[{subterm_no},{tech.tree.type},{v},{t},{period}]"
                            for tech in solve_node.heattransfertechNodeList for v in range(tech.NumVersion) for t in solve_node.allSubperiods if t <= period < t + tech.lifetime[v]])

                vars_to_fix = []
                values_to_fix = []
                for var_name in vars_subterm:
                    var = model.getVarByName(var_name)
                    vars_to_fix.append(var)
                    values_to_fix.append(var.X)
                model.setAttr('LB', vars_to_fix, values_to_fix)
                model.setAttr('UB', vars_to_fix, values_to_fix)

            prev_carry_values[solve_node_id] = {'e': solve_node.e_Carrying[period, numSubterms].X, 'h': solve_node.h_Carrying[period, numSubterms].X}

            del model

        print(f"Subperiod {period} completed in {time.time() - subperiod_start_time:.2f} seconds")
        print(f"Subperiod {period} optimization time {subperiod_optimization_time:.2f} seconds")

    csv_file.close()
    print(f"Total Execution Time: {time.time() - execution_start_time:.2f} seconds")