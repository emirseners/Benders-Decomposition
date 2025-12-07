from gurobipy import GRB, Model, quicksum, Env
import os
import math
import copy
import time
import concurrent.futures

class ScenarioNode:
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
        ancestor = self
        amount_subperiods = len(ancestor.stageSubperiods) 
        node1_stage = (t-1) // amount_subperiods
        node2_stage = (t_-1) // amount_subperiods
        ancestor_degree = node2_stage - node1_stage
        for _ in range(ancestor_degree):
            ancestor = ancestor.parent
        return ancestor

    def AddMasterDecisionVariables(self, model, continuous_flag):
        self.v_Plus = {}
        if continuous_flag:
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        else:
            continuous_tech_nodes = [x for x in self.techNodeList if x not in self.electricitygenerationtechNodeList]
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in continuous_tech_nodes for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
            self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.electricitygenerationtechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.INTEGER, name="plus_"+str(self.id))) # purchase

        if len(self.children) == 0:
            self.e_Charging = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycharge_"+str(self.id)) # inventory charge amount
            self.h_Charging = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcharge_"+str(self.id)) # inventory charge amount
            self.e_Discharging = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitydischarge_"+str(self.id)) # inventory discharge amount
            self.h_Discharging = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatdischarge_"+str(self.id)) # inventory discharge amount
            self.e_Satisfied = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricityused_"+str(self.id)) # electricity used from inventory and generation
            self.h_Satisfied = model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatused_"+str(self.id)) # heat used from inventory and generation
            self.y_Transfer = model.addVars([(p, tech.tree.type, v, t, self.stageSubperiods[-1]) for p in self.stageSubterms for tech in self.heattransfertechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= self.stageSubperiods[-1] < t + tech.lifetime[v]], lb=0, vtype=GRB.CONTINUOUS, name="transferredheat_"+str(self.id)) # electricity transfered to heat
            self.e_Carrying = {}
            self.e_Carrying.update(model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycarry_"+str(self.id))) # inventory carriage amount
            self.e_Carrying.update(model.addVars([(self.stageSubperiods[-2], self.numSubterms)], lb=0, vtype=GRB.CONTINUOUS, name="electricitycarry_"+str(self.id))) # inventory carriage amount
            self.h_Carrying = {}
            self.h_Carrying.update(model.addVars([(self.stageSubperiods[-1], p) for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcarry_"+str(self.id))) # inventory carriage amount
            self.h_Carrying.update(model.addVars([(self.stageSubperiods[-2], self.numSubterms)], lb=0, vtype=GRB.CONTINUOUS, name="heatcarry_"+str(self.id))) # inventory carriage amount

    def AddSubproblemDecisionVariables(self, model):
        self.v_Plus = model.addVars([(tech.tree.type, v, t)  for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], lb=0, vtype=GRB.CONTINUOUS, name="plus_"+str(self.id)) # purchase

        subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]

        self.e_Purchase = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitypurchase_"+str(self.id)) # electricity purchase amount
        self.h_Purchase = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatpurchase_"+str(self.id)) # heat purchase amount
        self.e_Charging = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycharge_"+str(self.id)) # inventory charge amount
        self.h_Charging = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcharge_"+str(self.id)) # inventory charge amount
        self.e_Discharging = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitydischarge_"+str(self.id)) # inventory discharge amount
        self.h_Discharging = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatdischarge_"+str(self.id)) # inventory discharge amount
        self.e_Satisfied = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricityused_"+str(self.id)) # electricity used from inventory and generation
        self.h_Satisfied = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatused_"+str(self.id)) # heat used from inventory and generation
        self.y_Transfer = model.addVars([(p, tech.tree.type, v, t, t_) for p in self.stageSubterms for tech in self.heattransfertechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in subperiods_of_interest if t <= t_ < t + tech.lifetime[v]], lb=0, vtype=GRB.CONTINUOUS, name="transferredheat_"+str(self.id)) # electricity transfered to heat
        self.e_Carrying = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="electricitycarry_"+str(self.id)) # inventory carriage amount
        self.h_Carrying = model.addVars([(t, p) for t in subperiods_of_interest for p in self.stageSubterms], lb=0, vtype=GRB.CONTINUOUS, name="heatcarry_"+str(self.id)) # inventory carriage amount

    def AddMasterObjectiveCoefficients(self, discount_factor):
        for tech in self.techNodeList:
            for v in range(tech.NumVersion):
                for t in self.stageSubperiods:
                    self.v_Plus[tech.tree.type,v,t].Obj = self.probability * tech.cost[v] * (discount_factor**(t)) + self.probability * tech.OMcost[v] * ((tech.OMcostchangebyyear[v])**(t)) * sum([discount_factor**(t_) for t_ in range(t, min(t + tech.lifetime[v], self.tree.numStages * self.tree.numSubperiods+1))])

    def AddSubproblemObjectiveCoefficients(self, electricity_purchasing_cost, heat_purchasing_cost, discount_factor):
        subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]
        for t in subperiods_of_interest:
            discount_t = discount_factor ** t
            e_cost_t = electricity_purchasing_cost[t] * discount_t
            h_cost_t = heat_purchasing_cost[t] * discount_t
            for p in self.stageSubterms:
                self.e_Purchase[t,p].Obj = e_cost_t
                self.h_Purchase[t,p].Obj = h_cost_t

    def AddMasterDemandConstraints(self, model, electricity_demand, heat_demand):
        if len(self.children) == 0:
            t_ = self.stageSubperiods[-1]
            for p, periodic_demand in enumerate(electricity_demand[t_]):
                model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p]*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*(1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.electricitygenerationtechNodeList) for v in range(self.electricitygenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].lifetime[v]) - self.e_Charging[t_, p+1] + self.e_Discharging[t_, p+1] - self.e_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                model.addConstr(quicksum(((-1/self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p])*self.y_Transfer[p+1,tech.tree.type,v,t,t_]) for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.e_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
            for p, periodic_demand in enumerate(heat_demand[t_]):
                model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p]*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*(1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.heatgenerationtechNodeList) for v in range(self.heatgenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].lifetime[v]) - self.h_Charging[t_, p+1] + self.h_Discharging[t_, p+1] - self.h_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                model.addConstr(quicksum(((1 - (self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].degradation_rate[v]*(t_ - t)))*self.y_Transfer[p+1,tech.tree.type,v,t,t_]) for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.h_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Heat_{t_}_{p}')

    def AddSubproblemDemandConstraints(self, model, electricity_demand, heat_demand):
        subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]
        if self.id != 0:
            for t_ in subperiods_of_interest:
                for p, periodic_demand in enumerate(electricity_demand[t_]):
                    model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].periodic_electricity[v][p]*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*(1 - (self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.electricitygenerationtechNodeList) for v in range(self.electricitygenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitygenerationtechNodeList[i].lifetime[v]) + self.e_Purchase[t_, p+1] - self.e_Charging[t_, p+1] + self.e_Discharging[t_, p+1] - self.e_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Electricity_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                    model.addConstr(quicksum(((-1/self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].periodic_heat_transfer_cop[v][p])*self.y_Transfer[p+1,tech.tree.type,v,t,t_]) for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.e_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
                for p, periodic_demand in enumerate(heat_demand[t_]):
                    model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].periodic_heat[v][p]*self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*(1 - (self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.heatgenerationtechNodeList) for v in range(self.heatgenerationtechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatgenerationtechNodeList[i].lifetime[v]) + self.h_Purchase[t_, p+1] - self.h_Charging[t_, p+1] + self.h_Discharging[t_, p+1] - self.h_Satisfied[t_, p+1] >= 0, name = f'N{self.id}_Heat_Demand_Met_by_Generation_Inventory_{t_}_{p}')
                    model.addConstr(quicksum(((1 - (self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].degradation_rate[v]*(t_ - t)))*self.y_Transfer[p+1,tech.tree.type,v,t,t_]) for i, tech in enumerate(self.heattransfertechNodeList) for v in range(self.heattransfertechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].lifetime[v]) + self.h_Satisfied[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Heat_{t_}_{p}')

    def AddMasterInventoryBalanceConstraints(self, model):
        if len(self.children) == 0:
            t_ = self.stageSubperiods[-1]
            for p in self.stageSubterms:
                if p == 1:
                    model.addConstr(self.e_Carrying[t_,p] - self.FindAncestorFromDiff(t_-1,t_).e_Carrying[t_-1, self.numSubterms] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                    model.addConstr(self.h_Carrying[t_,p] - self.FindAncestorFromDiff(t_-1,t_).h_Carrying[t_-1, self.numSubterms] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')
                else:
                    model.addConstr(self.e_Carrying[t_,p] - self.e_Carrying[t_,p-1] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                    model.addConstr(self.h_Carrying[t_,p] - self.h_Carrying[t_,p-1] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')

    def AddSubproblemInventoryBalanceConstraints(self, model):
        if self.id != 0:
            subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]
            for t_ in subperiods_of_interest:
                for p in self.stageSubterms:
                    if p == 1:
                        model.addConstr(self.e_Carrying[t_,p] - self.FindAncestorFromDiff(t_-1,t_).e_Carrying[t_-1, self.numSubterms] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                        model.addConstr(self.h_Carrying[t_,p] - self.FindAncestorFromDiff(t_-1,t_).h_Carrying[t_-1, self.numSubterms] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')
                    else:
                        model.addConstr(self.e_Carrying[t_,p] - self.e_Carrying[t_,p-1] - self.electricitystoragetechNodeList[0].storage_charging_efficiency[0] * self.e_Charging[t_,p] + (1 / self.electricitystoragetechNodeList[0].storage_discharging_efficiency[0]) * self.e_Discharging[t_,p] == 0 , name = f'N{self.id}_ElectricityInventoryBalance_{t_}_{p}')
                        model.addConstr(self.h_Carrying[t_,p] - self.h_Carrying[t_,p-1] - self.heatstoragetechNodeList[0].storage_charging_efficiency[0] * self.h_Charging[t_,p] + (1 / self.heatstoragetechNodeList[0].storage_discharging_efficiency[0]) * self.h_Discharging[t_,p] == 0 , name = f'N{self.id}_HeatInventoryBalance_{t_}_{p}')

    def AddMasterStorageCapacityConstraints(self, model):
        if len(self.children) == 0:
            t_ = self.stageSubperiods[-1]
            for p in self.stageSubterms:
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]) - self.e_Carrying[t_,p] >= 0, name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{p}')
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]) - self.h_Carrying[t_,p] >= 0, name = f'N{self.id}_HeatStorageCapacity_{t_}_{p}')

            t_ = self.stageSubperiods[-2]
            model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]) - self.e_Carrying[t_,self.numSubterms] >= 0, name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{self.numSubterms}')
            model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]) - self.h_Carrying[t_,self.numSubterms] >= 0, name = f'N{self.id}_HeatStorageCapacity_{t_}_{self.numSubterms}')

    def AddSubproblemStorageCapacityConstraints(self, model):
        subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]
        for t_ in subperiods_of_interest:
            for p in self.stageSubterms:
                if len(self.children) == 0 and t_ == self.stageSubperiods[-2] and p == self.numSubterms:
                    continue
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].electricity_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.electricitystoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).electricitystoragetechNodeList[i].lifetime[v]) - self.e_Carrying[t_,p] >= 0, name = f'N{self.id}_ElectricityStorageCapacity_{t_}_{p}')
                model.addConstr(quicksum(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].heat_storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.heatstoragetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).heatstoragetechNodeList[i].lifetime[v]) - self.h_Carrying[t_,p] >= 0, name = f'N{self.id}_HeatStorageCapacity_{t_}_{p}')

    def AddMasterHeatTransferCapacityConstraints(self, model):
        if len(self.children) == 0:
            t_ = self.stageSubperiods[-1]
            for i, tech in enumerate(self.heattransfertechNodeList):
                for v in range(tech.NumVersion):
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            for p in self.stageSubterms:
                                model.addConstr(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].heat_transfer_capacity[v] - self.y_Transfer[p,tech.tree.type,v,t,t_] >= 0, name = f'N{self.id}_Heat_Transfer_Capacity_{tech.tree.type}_{v}_{t}_{t_}_{p}')

    def AddSubproblemHeatTransferCapacityConstraints(self, model):
        subperiods_of_interest = self.stageSubperiods if len(self.children) != 0 else self.stageSubperiods[:-1]
        for t_ in subperiods_of_interest:
            for i, tech in enumerate(self.heattransfertechNodeList):
                for v in range(tech.NumVersion):
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            for p in self.stageSubterms:
                                model.addConstr(self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]*self.FindAncestorFromDiff(t,t_).heattransfertechNodeList[i].heat_transfer_capacity[v] - self.y_Transfer[p,tech.tree.type,v,t,t_] >= 0, name = f'N{self.id}_Heat_Transfer_Capacity_{tech.tree.type}_{v}_{t}_{t_}_{p}')

    def AddBudgetConstraints(self, model, budget):
        for t in self.stageSubperiods:
            if budget[t] is not None:
                model.addConstr(-quicksum(tech.cost[v] * self.v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion)) >= -budget[t], name = f'N{self.id}_Budget_{t}')

    def AddSpatialConstraints(self, model, spatial_limit):
        for t_ in self.stageSubperiods:
            if spatial_limit is not None:
                model.addConstr(-quicksum(tech.spatial_requirement[v] * self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t+tech.lifetime[v]) >= -spatial_limit, name = f'N{self.id}_Spatial_{t_}')

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

def Output(m):
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'} 
    status = m.status

    print('The optimization status is ' + status_code[status])
    if status == 2:    
        print('Optimal solution:')
        for v in m.getVars():
            if v.x > 0:
                print(str(v.varName) + " = " + str(v.x))    
        print('Optimal objective value: ' + str(m.objVal) + "\n")

def MasterProblemModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor, multi_cut_flag, scenario_paths, scenario_path_probabilities, continuous_flag, tolerance):
    model_key = 'MasterProblem'
    master_env = Env(empty=True)
    master_env.start()
    model = Model(model_key, env=master_env)

    for node in scenarioTree.nodes:
        node.AddMasterDecisionVariables(model, continuous_flag)
        node.AddMasterObjectiveCoefficients(discount_factor)
        node.AddMasterDemandConstraints(model, electricity_demand, heat_demand)
        node.AddMasterInventoryBalanceConstraints(model)
        node.AddMasterStorageCapacityConstraints(model)
        node.AddMasterHeatTransferCapacityConstraints(model)
        node.AddBudgetConstraints(model, budget)
        node.AddSpatialConstraints(model, spatial_limit=None)
        node.InitializeCurrentTech(initial_tech)
        node.AddUpperBoundsForIP(model, budget)
    
    if multi_cut_flag:
        theta = model.addVars(list(scenario_paths.keys()), lb=0, vtype=GRB.CONTINUOUS, name="theta")
        for sp_id, sp_probability in scenario_path_probabilities.items():
            theta[sp_id].Obj = sp_probability
    else:
        theta = model.addVar(lb=0, name="theta", vtype=GRB.CONTINUOUS)
        theta.Obj = 1

    log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')

    model.setParam('BarHomogeneous', 1)
    model.setParam('Threads', threads)
    model.setParam('LogFile', log_file_path)
    model.setParam('LogToConsole', 0)

    if not continuous_flag:
        model.setParam('MIPFocus', 3)    
        model.setParam('TimeLimit', 86400)
        model.setParam('MIPGap', tolerance)
        model.setParam('NodefileStart', 0.95)
        model.setParam('NodefileDir', '.')

    model.update()

    return model

def SubProblemModel(scenario_path_id, scenario_path_nodes, scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor):
    global _worker_model, _worker_env

    _worker_env = Env(empty=True)
    _worker_env.start()
    model_key = 'SubProblem' + str(scenario_path_id)
    _worker_model = Model(model_key, env=_worker_env)

    scenarioPathnodes = [scenarioTree.nodes[node_id] for node_id in scenario_path_nodes]

    for node in scenarioPathnodes:
        node.AddSubproblemDecisionVariables(_worker_model)
        node.AddSubproblemObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor)
        node.AddSubproblemDemandConstraints(_worker_model, electricity_demand, heat_demand)
        node.AddSubproblemInventoryBalanceConstraints(_worker_model)
        node.AddSubproblemStorageCapacityConstraints(_worker_model)
        node.AddSubproblemHeatTransferCapacityConstraints(_worker_model)

    log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')

    _worker_model.setParam('BarHomogeneous', 1)
    _worker_model.setParam('Threads', threads)
    _worker_model.setParam('LogFile', log_file_path)
    _worker_model.setParam('LogToConsole', 0)
    _worker_model.update()
    
    _worker_model._var_cache = {var.varName: var for var in _worker_model.getVars()}
    
    return _worker_model

def OperationalNonanticipativityModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads, discount_factor):
    model_key = 'OperationalNonanticipativity'
    operationalnonanticipativity_env = Env(empty=True)
    operationalnonanticipativity_env.start()
    model = Model(model_key, env=operationalnonanticipativity_env)

    for node in scenarioTree.nodes:
        if len(node.children) != 0:
            node.AddSubproblemDecisionVariables(model)
            node.AddSubproblemObjectiveCoefficients(electricity_purchasing_cost, heat_purchasing_cost, discount_factor)
            node.AddSubproblemDemandConstraints(model, electricity_demand, heat_demand)
            node.AddSubproblemInventoryBalanceConstraints(model)
            node.AddSubproblemStorageCapacityConstraints(model)
            node.AddSubproblemHeatTransferCapacityConstraints(model)

    log_file_path = os.path.join(results_directory, model_key + 'GurobiLog.txt')

    model.setParam('BarHomogeneous', 1)
    model.setParam('Threads', threads)
    model.setParam('LogFile', log_file_path)
    model.setParam('LogToConsole', 0)
    model.update()

    return model

def solve_subproblem(nonanticipativity_lookup):
    global _worker_model

    vars_to_fix = []
    bounds = []
    
    for var_name, fix_value in nonanticipativity_lookup.items():
        var = _worker_model._var_cache.get(var_name)
        if var is not None:
            vars_to_fix.append(var)
            bounds.append(fix_value)
    
    _worker_model.setAttr('LB', vars_to_fix, bounds)
    _worker_model.setAttr('UB', vars_to_fix, bounds)
    _worker_model.update()

    _worker_model.optimize()

    dv_coefficients = {}
    nonant_vars = set(nonanticipativity_lookup.keys())
    constant = _worker_model.objVal
    
    all_constrs = _worker_model.getConstrs()
    dual_values = _worker_model.getAttr('Pi', all_constrs)
    
    for constr, pi in zip(all_constrs, dual_values):
        if abs(pi) < 1e-12:
            continue
            
        row = _worker_model.getRow(constr)
        for i in range(row.size()):
            var = row.getVar(i)
            vname = var.varName
            
            if vname in nonant_vars:
                coeff = row.getCoeff(i)
                dv_coef = -coeff * pi
                dv_coefficients[vname] = dv_coefficients.get(vname, 0.0) + dv_coef
                constant -= dv_coef * nonanticipativity_lookup[vname]
    
    return _worker_model.objVal, constant, dv_coefficients

def add_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, scenario_path_probabilities, master_var_cache):
    constant_term = sum(subproblem_constants[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_probabilities.keys())
    theta_var = master_var_cache["theta"]
    cut_expr = theta_var - constant_term

    for sp_id, dict_of_dvs in subproblem_dv_coefficients.items():
        sp_prob = scenario_path_probabilities[sp_id]
        for dv_name, dv_coef in dict_of_dvs.items():
            cut_expr -= dv_coef * sp_prob * master_var_cache[dv_name]
    
    return cut_expr

def add_multiple_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, scenario_paths, master_var_cache):
    cut_exprs = {}
    
    for sp_id in scenario_paths.keys():
        theta_var = master_var_cache[f"theta[{sp_id}]"]
        cut_expr = theta_var - subproblem_constants[sp_id]
        
        for dv_name, dv_coef in subproblem_dv_coefficients[sp_id].items():
            cut_expr -= dv_coef * master_var_cache[dv_name]
        
        cut_exprs[sp_id] = cut_expr
    
    return cut_exprs

def write_cuts(cuts_file, iteration, subproblem_constants, subproblem_dv_coefficients, scenario_path_probabilities, multi_cut_flag):
    lines = ['-' * 30, f"Iteration {iteration}:"]
    
    if multi_cut_flag:
        for sp_id in scenario_path_probabilities.keys():
            parts = [f"theta[{sp_id}] >= {subproblem_constants[sp_id]:.3f}"]
            
            for dv_name, dv_coef in subproblem_dv_coefficients[sp_id].items():
                if abs(dv_coef) > 1e-10:
                    sign = '+' if dv_coef >= 0 else '-'
                    parts.append(f" {sign} {abs(dv_coef):.3f} * {dv_name}")
            
            lines.append(''.join(parts))
    else:
        constant_term = sum(subproblem_constants[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_probabilities.keys())
        parts = [f"theta >= {constant_term:.3f}"]
        
        aggregated_coeffs = {}
        for sp_id, dv_dict in subproblem_dv_coefficients.items():
            sp_prob = scenario_path_probabilities[sp_id]
            for dv_name, dv_coef in dv_dict.items():
                aggregated_coeffs[dv_name] = aggregated_coeffs.get(dv_name, 0.0) + dv_coef * sp_prob
        
        for dv_name, coef in aggregated_coeffs.items():
            if abs(coef) > 1e-10:
                sign = '+' if coef >= 0 else '-'
                parts.append(f" {sign} {abs(coef):.3f} * {dv_name}")
        
        lines.append(''.join(parts))
    
    cuts_file.write('\n'.join(lines) + '\n')
    cuts_file.flush()

def get_leaf_node_solution(leaf_node_id, leaf_parent_node_id, numStages, numSubperiods, numSubterms):
    global _worker_model
    leaf_vars = {}
    leaf_suffix = f'_{leaf_node_id}['

    exclude_vars = {
        f'electricitydischarge_{leaf_node_id}[{(numStages-1) * numSubperiods + 1},1]',
        f'heatdischarge_{leaf_node_id}[{(numStages-1) * numSubperiods + 1},1]'}
    
    for var in _worker_model.getVars():
        if leaf_suffix in var.varName and not var.varName.startswith('plus_') and var.varName not in exclude_vars:
            leaf_vars[var.varName] = var.X

    target_index = f'[{(numStages-1) * numSubperiods},{numSubterms}]'

    e_carry_var = _worker_model.getVarByName(f'electricitycarry_{leaf_parent_node_id}{target_index}')
    h_carry_var = _worker_model.getVarByName(f'heatcarry_{leaf_parent_node_id}{target_index}')
    
    return leaf_vars, (e_carry_var.varName, e_carry_var.X), (h_carry_var.varName, h_carry_var.X)

def write_final_subproblem_solutions(executors, nonanticipativity_lookup, results_directory, scenario_paths, numStages, numSubperiods, numSubterms):
    futures = {sp_id: executors[sp_id].submit(solve_subproblem, nonanticipativity_lookup) for sp_id in scenario_paths.keys()}
    
    for future in futures.values():
        future.result()
    
    leaf_futures = {}
    for sp_id, path_nodes in scenario_paths.items():
        leaf_node_id = path_nodes[-1]
        leaf_parent_node_id = path_nodes[-2]
        leaf_futures[sp_id] = executors[sp_id].submit(get_leaf_node_solution, leaf_node_id, leaf_parent_node_id, numStages, numSubperiods, numSubterms)
    
    electricity_carry_values = {}
    heat_carry_values = {}
    
    sol_filename = os.path.join(results_directory, 'Results.sol')
    with open(sol_filename, 'a') as f:
        lines = []
        for sp_id in scenario_paths.keys():
            leaf_vars, e_carry, h_carry = leaf_futures[sp_id].result()
            lines.extend(f'{var_name} {value}\n' for var_name, value in leaf_vars.items())

            var_name, value = e_carry
            if var_name not in electricity_carry_values or value > electricity_carry_values[var_name]:
                electricity_carry_values[var_name] = value
        
            var_name, value = h_carry
            if var_name not in heat_carry_values or value > heat_carry_values[var_name]:
                heat_carry_values[var_name] = value
        
        f.writelines(lines)
    
    return electricity_carry_values, heat_carry_values

def benders_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        iteration_start_time = time.time()
        call_back_data = model._callback_data
        call_back_data['iteration'] += 1

        nonant_values = model.cbGetSolution(call_back_data['nonant_vars'])
        nonanticipativity_lookup = dict(zip(call_back_data['nonant_var_names'], nonant_values))

        lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        subproblem_start_time = time.time()
        futures = {sp_id: call_back_data['executors'][sp_id].submit(solve_subproblem, nonanticipativity_lookup) for sp_id in call_back_data['scenario_paths'].keys()}
        
        subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
        subproblem_execution_time = time.time() - subproblem_start_time

        subproblem_objectives = {sp_id: result[0] for sp_id, result in subproblem_results.items()}
        subproblem_constants = {sp_id: result[1] for sp_id, result in subproblem_results.items()}
        subproblem_dv_coefficients = {sp_id: result[2] for sp_id, result in subproblem_results.items()}
        
        if call_back_data['multi_cut_flag']:
            if 'theta_vars' not in call_back_data:
                call_back_data['theta_vars'] = {sp_id: call_back_data['master_var_cache'][f"theta[{sp_id}]"] for sp_id in call_back_data['scenario_paths'].keys()}
            
            theta_sum = sum(model.cbGetSolution(call_back_data['theta_vars'][sp_id]) * sp_prob for sp_id, sp_prob in call_back_data['scenario_path_probabilities'].items())
            subproblem_obj_sum = sum(subproblem_objectives[sp_id] * call_back_data['scenario_path_probabilities'][sp_id] for sp_id in call_back_data['scenario_paths'].keys())
            upper_bound = current_obj - theta_sum + subproblem_obj_sum
            cut_expressions = add_multiple_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, call_back_data['scenario_paths'], call_back_data['master_var_cache'])
            for cut_expression in cut_expressions.values():
                model.cbLazy(cut_expression >= 0)
        else:
            if 'theta_var' not in call_back_data:
                call_back_data['theta_var'] = call_back_data['master_var_cache']["theta"]
            
            subproblem_obj_sum = sum(subproblem_objectives[sp_id] * call_back_data['scenario_path_probabilities'][sp_id] for sp_id in call_back_data['scenario_paths'].keys())
            upper_bound = current_obj - model.cbGetSolution(call_back_data['theta_var']) + subproblem_obj_sum
            cut_expression = add_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, call_back_data['scenario_path_probabilities'], call_back_data['master_var_cache'])
            model.cbLazy(cut_expression >= 0)

        if call_back_data['cuts_file']:
            write_cuts(call_back_data['cuts_file'], call_back_data['iteration'], subproblem_constants, subproblem_dv_coefficients, call_back_data['scenario_path_probabilities'], call_back_data['multi_cut_flag'])

        if upper_bound < call_back_data['best_upper_bound']:
            call_back_data['best_upper_bound'] = upper_bound
            all_vars = model.getVars()
            all_var_values = model.cbGetSolution(all_vars)
            call_back_data['best_ub_lookup'] = {var.varName: val for var, val in zip(all_vars, all_var_values) if not var.varName.startswith("theta")}

        call_back_data['final_lower_bound'] = lower_bound
        gap = (call_back_data['best_upper_bound'] - lower_bound) / max(1e-6, call_back_data['best_upper_bound'])
        
        log_lines = [
            '-' * 30,
            f"Iteration {call_back_data['iteration']}:",
            f"Upper Bound: {call_back_data['best_upper_bound']:.2f}",
            f"Lower Bound: {lower_bound:.2f}",
            f"Gap: {(100 * gap):.2f}%",
            f"Iteration Time: {time.time() - iteration_start_time:.2f} seconds",
            f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds"
        ]
        call_back_data['log_file'].write('\n'.join(log_lines) + '\n')
        call_back_data['log_file'].flush()

        if gap < call_back_data['tolerance']:
            model.terminate()

def CampusApplication(numStages, numSubperiods, numSubterms, scenarioTree, initial_tech, emission_limits, electricity_demand, heat_demand, budget,
                      electricity_purchasing_cost, heat_purchasing_cost, results_directory, discount_factor, scenario_paths, scenario_path_probabilities, 
                      tolerance, multi_cut_flag, callback_flag, incumbent_solution, write_cuts_flag, continuous_flag):
    execution_start_time = time.time()
    
    master_threads = 20
    threads_per_worker = 1
    
    executors = {}
    for scenario_path_id, scenario_path_nodes in scenario_paths.items():
        scenarioTree_copy = copy.deepcopy(scenarioTree)
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            initializer=SubProblemModel,
            initargs=(scenario_path_id, scenario_path_nodes, scenarioTree_copy, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads_per_worker, discount_factor)
        )
        executors[scenario_path_id] = executor

    master_model = MasterProblemModel(copy.deepcopy(scenarioTree), emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor, multi_cut_flag, scenario_paths, scenario_path_probabilities, continuous_flag, tolerance)

    if incumbent_solution is not None:
        for var_name, var_value in incumbent_solution.items():
            var = master_model.getVarByName(var_name)
            var.Start = var_value
        master_model.update()

    log_file = open(os.path.join(results_directory, 'BendersLog.txt'), 'w')
    cuts_file = open(os.path.join(results_directory, 'GeneratedCuts.txt'), 'w') if write_cuts_flag else None

    nonant_vars = [var for var in master_model.getVars() if not var.varName.startswith("theta")]
    master_var_cache = {var.varName: var for var in master_model.getVars()}

    if callback_flag:
        master_model.setParam('LazyConstraints', 1)
        nonant_var_names = [var.varName for var in nonant_vars]

        master_model._callback_data = {
            'iteration': 0,
            'log_file': log_file,
            'cuts_file': cuts_file,
            'executors': executors,
            'scenario_paths': scenario_paths,
            'scenario_path_probabilities': scenario_path_probabilities,
            'multi_cut_flag' : multi_cut_flag,
            'best_upper_bound': float('inf'),
            'final_lower_bound': None,
            'best_ub_lookup': None,
            'nonant_vars': nonant_vars,
            'nonant_var_names': nonant_var_names,
            'master_var_cache': master_var_cache,
            'tolerance': tolerance
        }

        master_model.optimize(benders_callback)
        iteration = master_model._callback_data['iteration']
        best_upper_bound = master_model._callback_data['best_upper_bound']
        lower_bound = master_model._callback_data['final_lower_bound']
        best_ub_lookup = master_model._callback_data['best_ub_lookup']

    else:
        iteration = 0
        max_iterations = 200
        best_upper_bound = float('inf')

        while iteration < max_iterations:
            iteration += 1

            master_start_time = time.time()
            master_model.optimize()
            master_execution_time = time.time() - master_start_time

            lower_bound = master_model.ObjVal

            nonant_solution_values = master_model.getAttr('X', nonant_vars)
            nonanticipativity_lookup = {var.varName: val for var, val in zip(nonant_vars, nonant_solution_values)}

            futures = {}
            subproblem_start_time = time.time()

            for sp_id in scenario_paths.keys():
                futures[sp_id] = executors[sp_id].submit(solve_subproblem, nonanticipativity_lookup)

            subproblem_results = {sp_id: futures[sp_id].result() for sp_id in futures.keys()}
            subproblem_execution_time = time.time() - subproblem_start_time

            subproblem_objectives = {sp_id: result[0] for sp_id, result in subproblem_results.items()}
            subproblem_constants = {sp_id: result[1] for sp_id, result in subproblem_results.items()}
            subproblem_dv_coefficients = {sp_id: result[2] for sp_id, result in subproblem_results.items()}

            if multi_cut_flag:
                upper_bound = master_model.ObjVal - sum([master_var_cache[f"theta[{sp_id}]"].X * sp_prob for sp_id, sp_prob in scenario_path_probabilities.items()])+ sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_paths.keys())
                cut_expressions = add_multiple_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, scenario_paths, master_var_cache)
                for sp_id, cut_expression in cut_expressions.items():
                    master_model.addConstr(cut_expression >= 0, name=f'OptimalityCut{sp_id}_{iteration}')
            else:
                upper_bound = master_model.ObjVal - master_var_cache["theta"].X + sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_paths.keys())
                cut_expression = add_optimality_cuts(subproblem_constants, subproblem_dv_coefficients, scenario_path_probabilities, master_var_cache)
                master_model.addConstr(cut_expression >= 0, name=f'OptimalityCut_{iteration}')

            if cuts_file:
                write_cuts(cuts_file, iteration, subproblem_constants, subproblem_dv_coefficients, scenario_path_probabilities, multi_cut_flag)

            if upper_bound < best_upper_bound:
                best_upper_bound = upper_bound
                best_ub_vars = master_model.getVars()
                best_ub_var_values = master_model.getAttr('X', best_ub_vars)
                best_ub_lookup = {var.varName: val for var, val in zip(best_ub_vars, best_ub_var_values) if not var.varName.startswith("theta")}
    
            gap = (best_upper_bound - lower_bound) / max(1e-6, best_upper_bound)
            log_lines = [
                '-' * 30,
                f"Iteration {iteration}:",
                f"Upper Bound: {best_upper_bound:.2f}",
                f"Lower Bound: {lower_bound:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Master Problem Execution Time: {master_execution_time:.2f} seconds",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds"
            ]
            log_file.write('\n'.join(log_lines) + '\n')
            log_file.flush()

            if tolerance > (best_upper_bound - lower_bound) / best_upper_bound:
                break

    final_gap = (best_upper_bound - lower_bound) / max(1e-6, best_upper_bound)
    summary_lines = [
        "=" * 30,
        "Final Summary",
        f"Total Iterations: {iteration}",
        f"Best Upper Bound: {best_upper_bound:.2f}",
        f"Final Lower Bound: {lower_bound:.2f}",
        f"Final Gap: {(100 * final_gap):.2f}%",
        f"Total Time: {time.time() - execution_start_time:.2f} seconds"
    ]
    log_file.write('\n'.join(summary_lines) + '\n')
    log_file.close()
    
    if cuts_file:
        cuts_file.close()
    
    final_sol_file = os.path.join(results_directory, 'Results.sol')
    with open(final_sol_file, 'w') as f:
        for var_name, var_value in best_ub_lookup.items():
            f.write(f"{var_name} {var_value}\n")
    
    electricity_carry_values, heat_carry_values = write_final_subproblem_solutions(executors, best_ub_lookup, results_directory, scenario_paths, numStages, numSubperiods, numSubterms)
    
    for executor in executors.values():
        executor.shutdown(wait=True)

    operational_model = OperationalNonanticipativityModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor)

    vars_to_fix = []
    bounds = []
    
    for var_name, var_value in best_ub_lookup.items():
        if var_name.startswith("plus_"):
            var = operational_model.getVarByName(var_name)
            if var is not None:
                vars_to_fix.append(var)
                bounds.append(var_value)
    
    operational_model.setAttr('LB', vars_to_fix, bounds)
    operational_model.setAttr('UB', vars_to_fix, bounds)

    for var_name, var_value in electricity_carry_values.items():
        var = operational_model.getVarByName(var_name)
        var.LB = var_value

    for var_name, var_value in heat_carry_values.items():
        var = operational_model.getVarByName(var_name)
        var.LB = var_value

    operational_model.update()
    operational_model.optimize()

    with open(final_sol_file, 'a') as f:
        for var in operational_model.getVars():
            if not var.varName.startswith('plus_'):
                f.write(f"{var.varName} {var.X}\n")

    sol_values = {}
    with open(final_sol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            sol_values[parts[0]] = float(parts[1])
    
    last_period = (numStages-1) * numSubperiods
    e_discharge_eff = scenarioTree.nodes[-1].electricitystoragetechNodeList[0].storage_discharging_efficiency[0]
    h_discharge_eff = scenarioTree.nodes[-1].heatstoragetechNodeList[0].storage_discharging_efficiency[0]
    
    discharge_lines = []
    for scenario_path_id, scenario_path_nodes in scenario_paths.items():
        leaf_node_id = scenario_path_nodes[-1]
        leaf_parent_node_id = scenario_path_nodes[-2]
        
        e_carry_prev = sol_values[f'electricitycarry_{leaf_parent_node_id}[{last_period},{numSubterms}]']
        e_carry_curr = sol_values[f'electricitycarry_{leaf_node_id}[{last_period + 1},1]']
        h_carry_prev = sol_values[f'heatcarry_{leaf_parent_node_id}[{last_period},{numSubterms}]']
        h_carry_curr = sol_values[f'heatcarry_{leaf_node_id}[{last_period + 1},1]']
        
        e_discharge = e_discharge_eff * (e_carry_prev - e_carry_curr)
        h_discharge = h_discharge_eff * (h_carry_prev - h_carry_curr)
        
        discharge_lines.append(f"electricitydischarge_{leaf_node_id}[{last_period + 1},1] {e_discharge}\n")
        discharge_lines.append(f"heatdischarge_{leaf_node_id}[{last_period + 1},1] {h_discharge}\n")
    
    with open(final_sol_file, 'a') as f:
        f.writelines(discharge_lines)