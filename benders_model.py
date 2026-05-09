from gurobipy import GRB, Model, quicksum, Env
import numpy as np
import math
import os

class ScenarioNode:
    def __init__(self, id_In, parent_In, probability_In, tree_In, techNodeList_In):
        self.id = id_In
        self.parent = parent_In
        self.tree = tree_In
        self._ancestor_cache = {}

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
        self.stageSubterms = list(range(1, self.numSubterms+1))
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
        child = ScenarioNode(len(self.tree.nodes), self, prob, self.tree, techNodeList)
        self.children.append(child)

    def FindAncestorFromDiff(self, t, t_):
        cache_key = (t, t_)
        cached = self._ancestor_cache.get(cache_key)
        if cached is not None:
            return cached
        
        ancestor = self
        amount_subperiods = self.numSubperiods
        node1_stage = (t-1) // amount_subperiods
        node2_stage = (t_-1) // amount_subperiods
        ancestor_degree = node2_stage - node1_stage
        for _ in range(ancestor_degree):
            ancestor = ancestor.parent
        
        self._ancestor_cache[cache_key] = ancestor
        return ancestor

    def AddMasterDecisionVariables(self, model, continuous_flag):
        self.v_Plus = {}
        if continuous_flag:
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        else:
            continuous_tech_nodes = [x for x in self.techNodeList if x not in self.electricitygenerationtechNodeList]
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in continuous_tech_nodes for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.electricitygenerationtechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.INTEGER, name="plus_"+str(self.id))) # purchase

    def AddSubproblemDecisionVariables(self, model):
        self.v_Plus = model.addVars([(tech.tree.type, v, t)  for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id)) # purchase

        if self.id != 0:
            self.e_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitypurchase_"+str(self.id)) # electricity purchase amount
            self.h_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatpurchase_"+str(self.id)) # heat purchase amount
            self.e_Charging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycharge_"+str(self.id)) # inventory charge amount
            self.h_Charging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcharge_"+str(self.id)) # inventory charge amount
            self.e_Discharging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitydischarge_"+str(self.id)) # inventory discharge amount
            self.h_Discharging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatdischarge_"+str(self.id)) # inventory discharge amount
            self.e_Satisfied = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricityused_"+str(self.id)) # electricity used from inventory and generation
            self.h_Satisfied = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatused_"+str(self.id)) # heat used from inventory and generation
            self.y_Transfer = model.addVars([(p, tech.tree.type, v, t, t_) for p in self.stageSubterms for tech in self.heattransfertechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], lb=0, vtype=GRB.CONTINUOUS, name="transferredheat_"+str(self.id)) # electricity transfered to heat
            self.e_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycarry_"+str(self.id)) # inventory carriage amount
            self.h_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcarry_"+str(self.id)) # inventory carriage amount
        else:
            self.e_Carrying = model.addVars([(t, self.numSubterms) for t in self.stageSubperiods], lb=0, ub=0, vtype=GRB.CONTINUOUS, name="electricitycarry_"+str(self.id)) # inventory carriage amount
            self.h_Carrying = model.addVars([(t, self.numSubterms) for t in self.stageSubperiods], lb=0, ub=0, vtype=GRB.CONTINUOUS, name="heatcarry_"+str(self.id)) # inventory carriage amount

    def AddMasterObjectiveCoefficients(self, discount_factor):
        for tech in self.techNodeList:
            for v in range(tech.NumVersion):
                for t in self.stageSubperiods:
                    self.v_Plus[tech.tree.type,v,t].Obj = 0.001 * self.probability * (tech.cost[v] * (discount_factor**(t)) + tech.OMcost[v] * ((tech.OMcostchangebyyear[v])**(t)) * sum([discount_factor**(t_) for t_ in range(t, min(t + tech.lifetime[v], self.tree.numStages * self.tree.numSubperiods+1))]))

    def AddSubproblemObjectiveCoefficients(self, electricity_purchasing_cost, heat_purchasing_cost, discount_factor, prob_flag):
        for t in self.stageSubperiods:
            discount_t = discount_factor ** t
            e_cost_t = electricity_purchasing_cost[t] * discount_t
            h_cost_t = heat_purchasing_cost[t] * discount_t
            if prob_flag:
                e_cost_t *= self.probability
                h_cost_t *= self.probability
            for p in self.stageSubterms:
                self.e_Purchase[t,p].Obj = 0.001 * e_cost_t
                self.h_Purchase[t,p].Obj = 0.001 * h_cost_t

    def AddSubproblemDemandConstraints(self, model, electricity_demand, heat_demand):
        for t_ in self.stageSubperiods:
            for p, periodic_demand in enumerate(electricity_demand[t_]):
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.electricitygenerationtechNodeList) for v in range(self.electricitygenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].lifetime[v]) + self.e_Purchase[t_, p+1] - self.e_Charging[t_, p+1] + self.e_Discharging[t_, p+1] - self.e_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                model.addConstr(quicksum((-1/self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p])*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.e_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
            for p, periodic_demand in enumerate(heat_demand[t_]):
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p]*(1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for i, tech in enumerate(self.heatgenerationtechNodeList) for v in range(self.heatgenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].lifetime[v]) + self.h_Purchase[t_, p+1] - self.h_Charging[t_, p+1] + self.h_Discharging[t_, p+1] - self.h_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                model.addConstr(quicksum((1 - (self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].degradation_rate[v]*(t_ - t)))*self.y_Transfer[p+1,tech.tree.type,v,t,t_] for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.h_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Heat_{t_}_{p}')

    def AddSubproblemInventoryBalanceConstraints(self, model):
        for t_ in self.stageSubperiods:
            for p in self.stageSubterms:
                if p == 1:
                    model.addConstr(self.e_Carrying[t_,p] - self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * self.FindAncestorFromDiff(t_-1,t_).e_Carrying[t_-1, self.numSubterms] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                    model.addConstr(self.h_Carrying[t_,p] - self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * self.FindAncestorFromDiff(t_-1,t_).h_Carrying[t_-1, self.numSubterms] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')
                else:
                    model.addConstr(self.e_Carrying[t_,p] - self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * self.e_Carrying[t_,p-1] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                    model.addConstr(self.h_Carrying[t_,p] - self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * self.h_Carrying[t_,p-1] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')

    def AddSubproblemStorageCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for p in self.stageSubterms:
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]) - self.e_Carrying[t_,p] >= 0, name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{p}')
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]) - self.h_Carrying[t_,p] >= 0, name = f'N{self.id}_HeatStorageCapacity_{t_}_{p}')

    def AddSubproblemHeatTransferCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for i, tech in enumerate(self.heattransfertechNodeList):
                for v in range(tech.NumVersion):
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            for p in self.stageSubterms:
                                model.addConstr(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].heat_transfer_capacity[v] - self.y_Transfer[p,tech.tree.type,v,t,t_] >= 0, name = f'N{self.id}_Heat_Transfer_Capacity_{tech.tree.type}_{v}_{t}_{t_}_{p}')

    def AddEmissionConstraints(self, model, emission_limits):
        for t in self.stageSubperiods:
            if emission_limits[t] is not None:
                model.addConstr(quicksum(self.e_Purchase[t,p] + self.h_Purchase[t,p] for p in self.stageSubterms) <= emission_limits[t], name = f'N{self.id}_Emission_{t}')

    def AddBudgetConstraints(self, model, budget):
        for t in self.stageSubperiods:
            if budget[t] is not None:
                model.addConstr(-quicksum(1e-6 * tech.cost[v] * self.v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion)) >= -1e-6 * budget[t], name = f'N{self.id}_Budget_{t}')

    def AddSpatialConstraints(self, model, spatial_limit):
        if spatial_limit is not None:
            for t_ in self.stageSubperiods:
                model.addConstr(-quicksum(tech.spatial_requirement[v] * self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t+tech.lifetime[v]) >= -spatial_limit, name = f'N{self.id}_Spatial_{t_}')

    def InitializeCurrentTech(self, initial_tech):
        if self.parent == None:
            for i, tech in enumerate(self.techNodeList):
                for v in range(tech.NumVersion):
                    if initial_tech[i][v] != 0:
                        self.v_Plus[tech.tree.type, v, 0].lb = initial_tech[i][v]
                        self.v_Plus[tech.tree.type, v, 0].ub = initial_tech[i][v]
                    else:
                        self.v_Plus[tech.tree.type, v, 0].ub = 0

    def AddUpperBoundsForIP(self, model, budget):
        for tech in self.electricitygenerationtechNodeList:
            for v in range(tech.NumVersion):
                for t in self.stageSubperiods:
                    if budget[t] is not None:
                        ub_v = math.floor(budget[t] / tech.cost[v])
                        self.v_Plus[tech.tree.type,v,t].ub = ub_v

    def ComputeSeparationData(self):
        t_ = self.stageSubperiods[-1]
        separation_data = {}

        elec_storage_var_names = []
        elec_storage_coefs = []
        for i, tech in enumerate(self.electricitystoragetechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_ + 1):
                    ancestor = self.FindAncestorFromDiff(t, t_)
                    if t <= t_ < t + ancestor.electricitystoragetechNodeList[i].lifetime[v]:
                        coeff = self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0] * self.electricitystoragetechNodeList[0].storage_self_discharge_rate[0] * ancestor.electricitystoragetechNodeList[i].electricity_storage_capacity[v] * (1 - (ancestor.electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t)))
                        elec_storage_var_names.append(f"plus_{ancestor.id}[" + tech.tree.type + f",{v},{t}]")
                        elec_storage_coefs.append(coeff)
        separation_data['elec_storage_var_names'] = elec_storage_var_names
        separation_data['elec_storage_coefs'] = elec_storage_coefs

        heat_storage_var_names = []
        heat_storage_coefs = []
        for i, tech in enumerate(self.heatstoragetechNodeList):
            for v in range(tech.NumVersion):
                for t in range(0, t_ + 1):
                    ancestor = self.FindAncestorFromDiff(t, t_)
                    if t <= t_ < t + ancestor.heatstoragetechNodeList[i].lifetime[v]:
                        coeff = self.heatstoragetechNodeList[0].storage_discharging_efficiency[0] * self.heatstoragetechNodeList[0].storage_self_discharge_rate[0] * ancestor.heatstoragetechNodeList[i].heat_storage_capacity[v] * (1 - (ancestor.heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t)))
                        heat_storage_var_names.append(f"plus_{ancestor.id}[" + tech.tree.type + f",{v},{t}]")
                        heat_storage_coefs.append(coeff)
        separation_data['heat_storage_var_names'] = heat_storage_var_names
        separation_data['heat_storage_coefs'] = heat_storage_coefs

        elec_gen_var_names = []
        elec_gen_coef_lists = []
        for i, tech in enumerate(self.electricitygenerationtechNodeList):
            for v in range(self.electricitygenerationtechNodeList[i].NumVersion):
                for t in range(0, t_ + 1):
                    ancestor = self.FindAncestorFromDiff(t, t_)
                    if t <= t_ < t + ancestor.electricitygenerationtechNodeList[i].lifetime[v]:
                        coeff_array = ancestor.electricitygenerationtechNodeList[i].periodic_electricity[v] * (1.0 - (ancestor.electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))
                        elec_gen_var_names.append(f"plus_{ancestor.id}[" + tech.tree.type + f",{v},{t}]")
                        elec_gen_coef_lists.append(coeff_array)
        elec_gen_coef_matrix = np.column_stack(elec_gen_coef_lists)
        separation_data['elec_gen_coef_matrix'] = elec_gen_coef_matrix
        separation_data['elec_coef_cumsum'] = np.vstack([np.zeros((1, elec_gen_coef_matrix.shape[1])), np.cumsum(elec_gen_coef_matrix, axis=0)])
        separation_data['elec_gen_var_names'] = elec_gen_var_names

        heat_gen_var_names = []
        heat_gen_coef_lists = []
        for i, tech in enumerate(self.heatgenerationtechNodeList):
            for v in range(self.heatgenerationtechNodeList[i].NumVersion):
                for t in range(0, t_ + 1):
                    ancestor = self.FindAncestorFromDiff(t, t_)
                    if t <= t_ < t + ancestor.heatgenerationtechNodeList[i].lifetime[v]:
                        coeff_array = ancestor.heatgenerationtechNodeList[i].periodic_heat[v] * (1.0 - (ancestor.heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))
                        heat_gen_var_names.append(f"plus_{ancestor.id}[" + tech.tree.type + f",{v},{t}]")
                        heat_gen_coef_lists.append(coeff_array)
        heat_gen_coef_matrix = np.column_stack(heat_gen_coef_lists)
        separation_data['heat_gen_coef_matrix'] = heat_gen_coef_matrix
        separation_data['heat_coef_cumsum'] = np.vstack([np.zeros((1, heat_gen_coef_matrix.shape[1])), np.cumsum(heat_gen_coef_matrix, axis=0)])
        separation_data['heat_gen_var_names'] = heat_gen_var_names

        heat_transfer_var_names = []
        heat_transfer_coefs = []
        for i, tech in enumerate(self.heattransfertechNodeList):
            for v in range(self.heattransfertechNodeList[i].NumVersion):
                for t in range(0, t_ + 1):
                    ancestor = self.FindAncestorFromDiff(t, t_)
                    if t <= t_ < t + ancestor.heattransfertechNodeList[i].lifetime[v]:
                        coeff = ancestor.heattransfertechNodeList[i].heat_transfer_capacity[v] * (1 - (ancestor.heattransfertechNodeList[i].degradation_rate[v] * (t_ - t)))
                        heat_transfer_var_names.append(f"plus_{ancestor.id}[" + tech.tree.type + f",{v},{t}]")
                        heat_transfer_coefs.append(coeff)
        separation_data['heat_transfer_var_names'] = heat_transfer_var_names
        separation_data['heat_transfer_coefs'] = heat_transfer_coefs

        return separation_data

def MasterProblemModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor, scenario_paths, scenario_path_probabilities, continuous_flag, valid_inequalities_flag, tolerance):
    model_key = 'MasterProblem'
    master_env = Env(empty=True)
    master_env.start()
    model = Model(model_key, env=master_env)

    for node in scenarioTree.nodes:
        node.AddMasterDecisionVariables(model, continuous_flag)
        node.InitializeCurrentTech(initial_tech)
        if node.id != 0:
            node.AddMasterObjectiveCoefficients(discount_factor)
            node.AddBudgetConstraints(model, budget)
            node.AddSpatialConstraints(model, spatial_limit=None)
            node.AddUpperBoundsForIP(model, budget)

    separation_data = None
    if valid_inequalities_flag:
        separation_data = {}
        elec_demand_val = np.array(electricity_demand[-1], dtype=np.float64)
        heat_demand_val = np.array(heat_demand[-1], dtype=np.float64)
        elec_demand_cumsum = np.concatenate(([0], np.cumsum(elec_demand_val)))
        heat_demand_cumsum = np.concatenate(([0], np.cumsum(heat_demand_val)))

        node_id_to_path_id = {node_id: pid for pid, nodes in scenario_paths.items() for node_id in nodes}
        for node in scenarioTree.nodes:
            if len(node.children) == 0:
                path_id = node_id_to_path_id[node.id]
                separation_data[path_id] = node.ComputeSeparationData()
                separation_data[path_id]['electricity_demand'] = elec_demand_val
                separation_data[path_id]['heat_demand'] = heat_demand_val
                separation_data[path_id]['electricity_demand_cumsum'] = elec_demand_cumsum
                separation_data[path_id]['heat_demand_cumsum'] = heat_demand_cumsum

    theta = model.addVars(list(scenario_paths.keys()), lb=0, vtype=GRB.CONTINUOUS, name="theta")
    for sp_id, sp_probability in scenario_path_probabilities.items():
        theta[sp_id].Obj = sp_probability

    #log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')
    #model.setParam('LogFile', log_file_path)
    #model.setParam('FeasibilityTol', 1e-8)
    model.setParam('Threads', threads)
    model.setParam('LogToConsole', 0)

    if not continuous_flag:
        model.setParam('MIPFocus', 3)
        model.setParam('TimeLimit', 86400)
        model.setParam('NodefileStart', 0.95)
        model.setParam('NodefileDir', '.')
        model.setParam('MIPGap', 0.5 * tolerance)

    model.update()

    #lp_filename = os.path.join(results_directory, f'{model_key}.lp')
    #model.write(lp_filename)

    return model, master_env, separation_data

def SubProblemModel(scenario_path_id, scenario_path_nodes, scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor, aggregated_subproblems_flag):
    global _worker_model, _worker_env

    _worker_env = Env(empty=True)
    _worker_env.start()
    model_key = 'SubProblem' + str(scenario_path_id)
    _worker_model = Model(model_key, env=_worker_env)

    scenarioPathnodes = [scenarioTree.nodes[node_id] for node_id in scenario_path_nodes]

    for node in scenarioPathnodes:
        node.AddSubproblemDecisionVariables(_worker_model)
        if node.id != 0:
            node.AddSubproblemObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor, aggregated_subproblems_flag)
            node.AddSubproblemDemandConstraints(_worker_model, electricity_demand, heat_demand)
            node.AddSubproblemInventoryBalanceConstraints(_worker_model)
            node.AddSubproblemStorageCapacityConstraints(_worker_model)
            node.AddSubproblemHeatTransferCapacityConstraints(_worker_model)
            node.AddEmissionConstraints(_worker_model, emission_limits)

    #log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')
    #_worker_model.setParam('LogFile', log_file_path)
    _worker_model.setParam('Threads', threads)
    _worker_model.setParam('LogToConsole', 0)
    _worker_model.setParam('InfUnbdInfo', 1)
    _worker_model.update()
    
    all_vars = _worker_model.getVars()
    all_constrs = _worker_model.getConstrs()
    var_name_to_idx = {var.varName: i for i, var in enumerate(all_vars)}

    master_vars = [var for var in all_vars if var.varName.startswith("plus_")]
    _worker_model._master_var_names = [var.varName for var in master_vars]

    A = _worker_model.getA()
    nonant_indices_list = [var_name_to_idx[name] for name in _worker_model._master_var_names]
    _worker_model._A_master = A[:, nonant_indices_list].tocsr()
    _worker_model._all_rhs = np.array([constr.RHS for constr in all_constrs], dtype=np.float64)

    coupling_indices = np.unique(_worker_model._A_master.nonzero()[0])
    _worker_model._A_coupling = _worker_model._A_master[coupling_indices, :].tocsr()
    _worker_model._coupling_rhs = _worker_model._all_rhs[coupling_indices]

    _worker_model.remove(master_vars)
    _worker_model.update()
    _worker_model._all_constrs = _worker_model.getConstrs()
    _worker_model._coupling_constrs = [_worker_model._all_constrs[i] for i in coupling_indices]

    return _worker_model

def OperationalNonanticipativityModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor, aggregated_subproblems_flag):
    model_key = 'OperationalNonanticipativity'
    operationalnonanticipativity_env = Env(empty=True)
    operationalnonanticipativity_env.start()
    model = Model(model_key, env=operationalnonanticipativity_env)

    for node in scenarioTree.nodes:
        if len(node.children) != 0:
            node.AddSubproblemDecisionVariables(model)
            if node.id != 0:
                node.AddSubproblemObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor, True)
                node.AddSubproblemDemandConstraints(model, electricity_demand, heat_demand)
                node.AddSubproblemInventoryBalanceConstraints(model)
                node.AddSubproblemStorageCapacityConstraints(model)
                node.AddSubproblemHeatTransferCapacityConstraints(model)

    #log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')
    #model.setParam('LogFile', log_file_path)
    model.setParam('Threads', threads)
    model.setParam('LogToConsole', 0)
    model.update()

    return model, operationalnonanticipativity_env