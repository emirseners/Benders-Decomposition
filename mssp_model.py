from gurobipy import GRB, Model, quicksum
import os
import math
from fetch_data import fetch_data
from scenario_tree import generate_scenario_tree

class ScenarioNodeMSSP:
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
        child = ScenarioNodeMSSP(len(self.tree.nodes), self, prob, self.tree, techNodeList)
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

    def AddVariables(self, model):
        continuous_tech_nodes = [x for x in self.techNodeList if x not in self.electricitygenerationtechNodeList]
        self.v_Plus = {}
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in continuous_tech_nodes for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.electricitygenerationtechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.INTEGER, name="plus_"+str(self.id))) # purchase
        self.e_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="electricitypurchase_"+str(self.id)) # electricity purchase amount
        self.h_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="heatpurchase_"+str(self.id)) # heat purchase amount
        self.e_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="electricitycarry_"+str(self.id)) # inventory carriage amount
        self.h_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="heatcarry_"+str(self.id)) # inventory carriage amount
        self.e_Charging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="electricitycharge_"+str(self.id)) # inventory charge amount
        self.h_Charging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="heatcharge_"+str(self.id)) # inventory charge amount
        self.e_Discharging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="electricitydischarge_"+str(self.id)) # inventory discharge amount
        self.h_Discharging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="heatdischarge_"+str(self.id)) # inventory discharge amount
        self.e_Satisfied = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="electricityused_"+str(self.id)) # electricity used from inventory and generation
        self.h_Satisfied = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, lb=0, name="heatused_"+str(self.id)) # heat used from inventory and generation
        self.y_Transfer = model.addVars([(p, tech.tree.type, v, t, t_) for p in self.stageSubterms for tech in self.heattransfertechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, lb=0, name="transferredheat_"+str(self.id)) # electricity transfered to heat

    def AddObjectiveCoefficients(self, electricity_purchasing_cost, heat_purchasing_cost, discount_factor):
        if self.id != 0:
            for tech in self.techNodeList:
                for v in range(tech.NumVersion):
                    for t in self.stageSubperiods:
                        self.v_Plus[tech.tree.type,v,t].Obj = self.probability * tech.cost[v] * (discount_factor**(t)) + self.probability * tech.OMcost[v] * ((tech.OMcostchangebyyear[v])**(t)) * sum([discount_factor**(t_) for t_ in range(t, min(t + tech.lifetime[v], self.tree.numStages * self.tree.numSubperiods+1))])

            for t in self.stageSubperiods:
                for p in self.stageSubterms:
                    self.e_Purchase[t,p].Obj = self.probability * electricity_purchasing_cost[t] * (discount_factor**(t))
                    self.h_Purchase[t,p].Obj = self.probability * heat_purchasing_cost[t] * (discount_factor**(t))

    def AddDemandConstraints(self, model, electricity_demand, heat_demand):
        if self.id != 0:
            for t_ in self.stageSubperiods:
                for p, periodic_demand in enumerate(electricity_demand[t_]):
                    model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.electricitygenerationtechNodeList) for v in range(self.electricitygenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].lifetime[v]) + self.e_Purchase[t_, p+1] - self.e_Charging[t_, p+1] + self.e_Discharging[t_, p+1] >= self.e_Satisfied[t_, p+1], name = f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                    model.addConstr(quicksum(-1/self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p]*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.e_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
                for p, periodic_demand in enumerate(heat_demand[t_]):
                    model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.heatgenerationtechNodeList) for v in range(self.heatgenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].lifetime[v]) + self.h_Purchase[t_, p+1] - self.h_Charging[t_, p+1] + self.h_Discharging[t_, p+1] >= self.h_Satisfied[t_, p+1], name = f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                    model.addConstr(quicksum(1 - (self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].degradation_rate[v]*(t_ - t))*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.h_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Heat_{t_}_{p}')

    def AddInventoryBalanceConstraints(self, model):   #Global charging/discharging efficiencies are used.
        if self.id != 0:
            for t_ in self.stageSubperiods:
                for p in self.stageSubterms:
                    if p == 1:
                        model.addConstr(self.e_Carrying[t_,p] == self.FindAncestorFromDiff(t_-1,t_).e_Carrying[t_-1, self.numSubterms] + self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] - (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p], name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                        model.addConstr(self.h_Carrying[t_,p] == self.FindAncestorFromDiff(t_-1,t_).h_Carrying[t_-1, self.numSubterms] + self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] - (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p], name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')
                    else:
                        model.addConstr(self.e_Carrying[t_,p] == self.e_Carrying[t_,p-1] + self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] - (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p], name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                        model.addConstr(self.h_Carrying[t_,p] == self.h_Carrying[t_,p-1] + self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] - (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p], name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')

    def AddStorageCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for p in self.stageSubterms:
                model.addConstr(self.e_Carrying[t_,p] <= quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]), name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{p}')
                model.addConstr(self.h_Carrying[t_,p] <= quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]), name = f'N{self.id}_HeatStorageCapacity_{t_}_{p}')

    def AddHeatTransferCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for i, tech in enumerate(self.heattransfertechNodeList):
                for v in range(tech.NumVersion):
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            for p in self.stageSubterms:
                                model.addConstr(self.y_Transfer[p,tech.tree.type,v,t,t_] <= self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].heat_transfer_capacity[v], name = f'N{self.id}_Heat_Transfer_Capacity_{tech.tree.type}_{v}_{t}_{t_}_{p}')

    def AddEmissionConstraints(self, model, emission_limits):
        for t in self.stageSubperiods:
            if emission_limits[t] is not None:
                model.addConstr(quicksum(self.e_Purchase[t,p] + self.h_Purchase[t,p] for p in self.stageSubterms) <= emission_limits[t], name = f'N{self.id}_Emission_{t}')

    def AddBudgetConstraints(self, model, budget):
        for t in self.stageSubperiods:
            if budget[t] is not None:
                model.addConstr(quicksum(0.01 * tech.cost[v] * self.v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion)) <= 0.01 * budget[t], name = f'N{self.id}_Budget_{t}')

    def AddSpatialConstraints(self, model, spatial_limit):
        for t_ in self.stageSubperiods:
            if spatial_limit is not None:
                model.addConstr(quicksum(tech.spatial_requirement[v] * self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t+tech.lifetime[v]) <= spatial_limit, name = f'N{self.id}_Spatial_{t_}')

    def InitializeCurrentTech(self, initial_tech):
        if self.parent == None:
            for i, tech in enumerate(self.techNodeList):
                for v in range(tech.NumVersion):
                    if initial_tech[i][v] != 0:
                        self.v_Plus[tech.tree.type, v, 0].lb = initial_tech[i][v]
                    else:
                        self.v_Plus[tech.tree.type, v, 0].ub = 0

    def AddUpperBoundsForIP(self, model, budget):
        if self.id != 0:
            for tech in self.electricitygenerationtechNodeList:
                for v in range(tech.NumVersion):
                    for t in self.stageSubperiods:
                        if budget[t] is not None:
                            ub_v = math.floor(budget[t] / tech.cost[v])
                            model.addConstr(ub_v >= self.v_Plus[tech.tree.type,v,t], name = f'N{self.id}_UpperBound_v_plus_{tech.tree.type}_{v}_{t}')

def MSSPProblemModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, discount_factor, results_sol_path, tolerance, model_name):
    model = Model(model_name)
    model.setParam('OutputFlag', True)

    for node in scenarioTree.nodes:
        node.AddVariables(model)
        node.AddObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor)
        node.AddDemandConstraints(model, electricity_demand, heat_demand)
        node.AddInventoryBalanceConstraints(model)
        node.AddStorageCapacityConstraints(model)
        node.AddHeatTransferCapacityConstraints(model)
        node.AddBudgetConstraints(model, budget)
        node.AddEmissionConstraints(model, emission_limits)
        node.InitializeCurrentTech(initial_tech)
        node.AddSpatialConstraints(model, spatial_limit=None)
        node.AddUpperBoundsForIP(model, budget)

    model.update()

    if results_sol_path:
        solution_values = {}
        with open(results_sol_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    var_name = parts[0]
                    var_value = float(parts[1])
                    solution_values[var_name] = var_value

        vars_to_fix = []
        values_to_fix = []
        unfixed_vars = []

        for var in model.getVars():
            if var.varName in solution_values:
                vars_to_fix.append(var)
                values_to_fix.append(solution_values[var.varName])
            else:
                unfixed_vars.append(var.varName)

        model.setAttr('LB', vars_to_fix, values_to_fix)
        model.setAttr('UB', vars_to_fix, values_to_fix)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    log_file_path = os.path.join(results_directory, f'{model_name}_GurobiLog.txt')

    model.setParam('MIPFocus', 3)    
    model.setParam('TimeLimit', 86400)
    model.setParam('MIPGap', tolerance)
    model.setParam('NodefileStart', 0.95)
    model.setParam('Threads', 4)
    model.setParam('LogFile', log_file_path)
    model.setParam('LogToConsole', 0)
    model.setParam('NodefileDir', '.')
    model.update()
    model.optimize()

    lp_filename = os.path.join(results_directory, f'{model_name}.lp')
    model.write(lp_filename)

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        iis_file_path = os.path.join(results_directory, f'{model_name}_IIS.ilp')
        model.write(iis_file_path)

    return model

if __name__ == '__main__':
    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    numMultipliers = 2
    tolerance = 0.01

    input_data = fetch_data(numStages, numSubperiods, numSubterms)

    results_sol_path = os.path.join(input_data['results_directory'], 'Results.sol')
    
    scenario_tree_verify, initial_tech_verify = generate_scenario_tree(input_data['solar_initial'], input_data['solar_periodic_generation'], input_data['solar_advancements'], input_data['wind_initial'], input_data['wind_periodic_generation'], input_data['wind_advancements'], input_data['electricity_storage_initial'], input_data['electricity_storage_advancements'], input_data['parabolic_trough_initial'], input_data['parabolic_trough_periodic_generation'], input_data['parabolic_trough_advancements'], input_data['heat_pump_initial'], input_data['heat_pump_cop'], input_data['heat_pump_advancements'], input_data['heat_storage_initial'], input_data['heat_storage_advancements'], numSubterms, numSubperiods, numStages, numMultipliers, mssp_flag=True)
    
    MSSPProblemModel(scenario_tree_verify, input_data['emission_limits'], input_data['electricity_demand'], input_data['heat_demand'], 
                          initial_tech_verify, input_data['budget'], input_data['electricity_purchasing_cost'], input_data['heat_purchasing_cost'], 
                          input_data['results_directory'], input_data['discount_factor'], results_sol_path, tolerance, model_name='VerifyFeasibility')