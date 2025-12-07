import itertools

class ScenarioTree:
    def __init__(self, technologyTrees, mssp_flag=False):
        self.technologies = technologyTrees
        self.numStages = self.technologies[0].nodes[-1].stage
        self.numSubperiods = self.technologies[0].nodes[-1].numSubperiods
        self.numSubterms = self.technologies[0].numSubterms
        self.nodes = []

        if mssp_flag:
            from mssp_model import ScenarioNodeMSSP
            NodeClass = ScenarioNodeMSSP
        else:
            from benders_decomposition import ScenarioNode
            NodeClass = ScenarioNode

        prerootTechNodeList = []
        for tech in self.technologies:
            prerootTechNodeList.append(tech.nodes[0])
        
        self.preroot = NodeClass(0, None, 1, self, prerootTechNodeList)

        rootTechNodeList = []
        for tech in self.technologies:
            rootTechNodeList.append(tech.nodes[1])
        self.root = NodeClass(1, self.preroot, 1, self, rootTechNodeList)
        self.preroot.children.append(self.root)

        self.leaves = [self.root]
        for n in range(1,self.numStages):
            nextLeaves = []
            for leaf in self.leaves:
                tempList = []
                for techNode in leaf.techNodeList:
                    tempList.append(techNode.children)
                tempProduct = itertools.product(*tempList)
                for element in tempProduct:
                    leaf.AddChild(element)
                for child in leaf.children:
                    nextLeaves.append(child)
            self.leaves = []
            for leaf in nextLeaves:
                self.leaves.append(leaf)


class TechnologyTree:
    def __init__(self, type, segment, numSubperiods, numSubterms, lifetime, initialCost, degradation_rate, initialOMcost, depreciation_rate, OMcostchangebyyear, spatial_requirement, periodic_electricity_generation=None, periodic_heat_generation=None, electricity_storage_capacity=None, heat_storage_capacity=None, heat_transfer_capacity=None, periodic_heat_transfer_cop=None, storage_charging_efficiency=None, storage_discharging_efficiency=None):
        self.type = type
        self.initialCost = initialCost
        self.periodic_electricity_generation = periodic_electricity_generation
        self.periodic_heat_generation = periodic_heat_generation
        self.electricity_storage_capacity = electricity_storage_capacity
        self.heat_storage_capacity = heat_storage_capacity
        self.heat_transfer_capacity = heat_transfer_capacity
        self.periodic_heat_transfer_cop = periodic_heat_transfer_cop
        self.storage_charging_efficiency = storage_charging_efficiency
        self.storage_discharging_efficiency = storage_discharging_efficiency
        self.nodes = []
        self.degradation_rate = degradation_rate
        self.initialOMcost = initialOMcost
        self.OMcostchangebyyear = OMcostchangebyyear
        self.depreciation_rate = depreciation_rate
        self.lifetime = lifetime
        self.segment = segment
        self.spatial_requirement = spatial_requirement
        self.numSubperiods = numSubperiods
        self.numSubterms = numSubterms
        self.versions = len(initialCost)
        self.preroot = TechnologyNode(0, None, 1, self, self.versions, [0 for _ in range(self.versions)], self.periodic_electricity_generation, self.periodic_heat_generation, self.electricity_storage_capacity, self.heat_storage_capacity, self.heat_transfer_capacity, self.periodic_heat_transfer_cop, self.storage_charging_efficiency, self.storage_discharging_efficiency, self.lifetime, self.degradation_rate, self.initialOMcost, self.depreciation_rate, self.OMcostchangebyyear, self.spatial_requirement) # preroot is stage-0: the existing situation
        self.root = TechnologyNode(1, self.preroot, 1, self, self.versions, self.initialCost, self.periodic_electricity_generation, self.periodic_heat_generation, self.electricity_storage_capacity, self.heat_storage_capacity, self.heat_transfer_capacity, self.periodic_heat_transfer_cop, self.storage_charging_efficiency, self.storage_discharging_efficiency, self.lifetime, self.degradation_rate, self.initialOMcost, self.depreciation_rate, self.OMcostchangebyyear, self.spatial_requirement) #root is the inital decision making node.

    def ConstructByMultipliers(self, numStages, probabilities, costMultiplier, efficiencyMultiplier):
        leaves = [self.root]
        numBranches = len(probabilities)
        for n in range(1,numStages):
            nextLeaves = []
            for leaf in leaves:
                for b in range(numBranches):
                    leaf.AddChild(probabilities[b], costMultiplier[b], efficiencyMultiplier[b])
                for child in leaf.children:
                    nextLeaves.append(child)
            leaves = []
            for leaf in nextLeaves:
                leaves.append(leaf)

class TechnologyNode:
    def __init__(self, id_In, parent_In, probability_In, tree_In, versionnum_In, cost_In, periodic_electricity_In, periodic_heat_In, electricity_storage_capacity_In, heat_storage_capacity_In, heat_transfer_capacity_In, periodic_heat_transfer_cop_In, storage_charging_efficiency_In, storage_discharging_efficiency_In, lifetime_In, degradation_In, OMcost_In, depreciation_In, OMcostchangebyyear_In, spatial_requirement_In):
        self.id = id_In
        self.parent = parent_In

        self.tree = tree_In
        self.tree.nodes.append(self)
        self.children = []

        if self.parent is None:
            self.stage = 0
            self.probability = 1
            self.numSubperiods = 1
        else:
            self.stage = self.parent.stage + 1
            self.probability = self.parent.probability * probability_In
            self.numSubperiods = self.tree.numSubperiods

        self.NumVersion = versionnum_In
        self.cost = cost_In
        self.lifetime = lifetime_In
        self.periodic_electricity = periodic_electricity_In
        self.periodic_heat = periodic_heat_In
        self.electricity_storage_capacity = electricity_storage_capacity_In
        self.heat_storage_capacity = heat_storage_capacity_In
        self.heat_transfer_capacity = heat_transfer_capacity_In
        self.periodic_heat_transfer_cop = periodic_heat_transfer_cop_In
        self.storage_charging_efficiency = storage_charging_efficiency_In
        self.storage_discharging_efficiency = storage_discharging_efficiency_In
        self.degradation_rate = degradation_In
        self.OMcost = OMcost_In
        self.OMcostchangebyyear = OMcostchangebyyear_In
        self.depreciation_rate = depreciation_In
        self.spatial_requirement = spatial_requirement_In

    def AddChild(self, prob, costMult, effMult):
        child = TechnologyNode(len(self.tree.nodes), self, prob, self.tree, self.NumVersion, [i*costMult for i in self.cost], ([[x*effMult for x in i] for i in self.periodic_electricity] if self.periodic_electricity is not None else None), ([[x*effMult for x in i] for i in self.periodic_heat] if self.periodic_heat is not None else None), ([i*effMult for i in self.electricity_storage_capacity] if self.electricity_storage_capacity is not None else None), ([i*effMult for i in self.heat_storage_capacity] if self.heat_storage_capacity is not None else None), ([i*effMult for i in self.heat_transfer_capacity] if self.heat_transfer_capacity is not None else None), ([[x*effMult for x in i] for i in self.periodic_heat_transfer_cop] if self.periodic_heat_transfer_cop is not None else None), self.storage_charging_efficiency, self.storage_discharging_efficiency, self.lifetime, self.degradation_rate, self.OMcost, self.depreciation_rate, self.OMcostchangebyyear, self.spatial_requirement)
        self.children.append(child)


def extract_dataframe_parameters(df, row_index):
    return [df.iloc[row_index, i] for i in range(1, df.shape[1])]


def create_advancement_parameters(advancements_dict, num_multipliers, parameter_row):
    return [advancements_dict[num_multipliers][col][parameter_row] for col in advancements_dict[num_multipliers].columns if col != "Metrics"]


def create_technology_tree(tech_type, segment, initial_data, periodic_data, num_subperiods, num_subterms, **kwargs):
    base_params = {
        'initialCost': extract_dataframe_parameters(initial_data, 0),
        'lifetime': extract_dataframe_parameters(initial_data, 1),
        'degradation_rate': extract_dataframe_parameters(initial_data, 2),
        'initialOMcost': extract_dataframe_parameters(initial_data, 3),
        'OMcostchangebyyear': extract_dataframe_parameters(initial_data, 4),
        'depreciation_rate': extract_dataframe_parameters(initial_data, 5),
        'spatial_requirement': extract_dataframe_parameters(initial_data, 6)
    }

    if periodic_data is not None:
        processed_periodic_data = [[max(0, x) for x in sublist[:num_subterms]] for sublist in periodic_data]
        if 'electricity' in segment:
            base_params['periodic_electricity_generation'] = processed_periodic_data
        elif 'heat' in segment:
            base_params['periodic_heat_generation'] = processed_periodic_data
    
    base_params.update(kwargs)
    
    return TechnologyTree(tech_type, segment, num_subperiods, num_subterms, **base_params)


def construct_technology_with_multipliers(technology, num_stages, advancements_dict, advancement_key):
    technology.ConstructByMultipliers(numStages=num_stages,
        probabilities=create_advancement_parameters(advancements_dict, advancement_key, 0),
        costMultiplier=create_advancement_parameters(advancements_dict, advancement_key, 3),
        efficiencyMultiplier=create_advancement_parameters(advancements_dict, advancement_key, 4))


def generate_scenario_tree(solar_initial, solar_periodic_generation, solar_advancements, wind_initial, wind_periodic_generation, wind_advancements, electricity_storage_initial, 
                           electricity_storage_advancements, parabolic_trough_initial, parabolic_trough_periodic_generation, parabolic_trough_advancements, heat_pump_initial,
                           heat_pump_cop, heat_pump_advancements, heat_storage_initial, heat_storage_advancements, numSubterms, numSubperiods, numStages, numMultipliers, mssp_flag=False):

    solar = create_technology_tree('solar', 'electricity generation', solar_initial, solar_periodic_generation, numSubperiods, numSubterms)
    construct_technology_with_multipliers(solar, numStages, solar_advancements, numMultipliers)

    wind = create_technology_tree('wind', 'electricity generation', wind_initial, wind_periodic_generation, numSubperiods, numSubterms)
    construct_technology_with_multipliers(wind, numStages, wind_advancements, 1)

    electricity_storage = create_technology_tree('electricity_storage', 'electricity storage', electricity_storage_initial, None, numSubperiods, numSubterms,
        electricity_storage_capacity=extract_dataframe_parameters(electricity_storage_initial, 8),
        storage_charging_efficiency=extract_dataframe_parameters(electricity_storage_initial, 9),
        storage_discharging_efficiency=extract_dataframe_parameters(electricity_storage_initial, 10))
    construct_technology_with_multipliers(electricity_storage, numStages, electricity_storage_advancements, numMultipliers)

    parabolic_trough = create_technology_tree('parabolic_trough', 'heat generation', parabolic_trough_initial, parabolic_trough_periodic_generation, numSubperiods, numSubterms)
    construct_technology_with_multipliers(parabolic_trough, numStages, parabolic_trough_advancements, 1)

    heat_pump = create_technology_tree('heat_pump', 'heat transfer', heat_pump_initial, None, numSubperiods, numSubterms,
        heat_transfer_capacity=[int((8760-24)/numSubterms) * heat_pump_initial.iloc[8, i] for i in range(1, heat_pump_initial.shape[1])],
        periodic_heat_transfer_cop=[[max(0, x) for x in sublist[:numSubterms]] for sublist in heat_pump_cop])
    construct_technology_with_multipliers(heat_pump, numStages, heat_pump_advancements, 1)

    heat_storage = create_technology_tree('heat_storage', 'heat storage', heat_storage_initial, None, numSubperiods, numSubterms,
        heat_storage_capacity=extract_dataframe_parameters(heat_storage_initial, 8),
        storage_charging_efficiency=extract_dataframe_parameters(heat_storage_initial, 9),
        storage_discharging_efficiency=extract_dataframe_parameters(heat_storage_initial, 10))
    construct_technology_with_multipliers(heat_storage, numStages, heat_storage_advancements, 1)

    initial_tech = [
        extract_dataframe_parameters(solar_initial, 7),
        extract_dataframe_parameters(wind_initial, 7),
        extract_dataframe_parameters(electricity_storage_initial, 7),
        extract_dataframe_parameters(parabolic_trough_initial, 7),
        extract_dataframe_parameters(heat_pump_initial, 7),
        extract_dataframe_parameters(heat_storage_initial, 7)
    ]

    scenarioTree = ScenarioTree([solar, wind, electricity_storage, parabolic_trough, heat_pump, heat_storage], mssp_flag)

    return scenarioTree, initial_tech