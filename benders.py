from gurobipy import GRB, Model, quicksum, Env
import os
import copy
import time
import threading
import concurrent.futures
from collections import defaultdict

_cached_worker_model = None

def _init_worker_subproblem(subproblem_builder, *args):
    global _cached_worker_model
    _cached_worker_model = subproblem_builder(*args)

def solve_subproblem(nonanticipativity_lookup):
    _worker_model = _cached_worker_model

    nonant_vars = _worker_model._nonant_vars
    nonant_var_names = _worker_model._nonant_var_names
    bounds = [nonanticipativity_lookup[name] for name in nonant_var_names]
    
    _worker_model.setAttr('LB', nonant_vars, bounds)
    _worker_model.setAttr('UB', nonant_vars, bounds)
    
    _worker_model.optimize()

    status = _worker_model.status
    feasibility_flag = status == GRB.OPTIMAL

    var_name_to_idx = _worker_model._var_name_to_idx
    constr_nonant_map = _worker_model._constr_nonant_map
    all_constrs = _worker_model._all_constrs
    nonant_idx_to_name = _worker_model._nonant_idx_to_name
    
    nonant_values = {}
    for name, val in nonanticipativity_lookup.items():
        idx = var_name_to_idx.get(name)
        if idx is not None:
            nonant_values[idx] = val
    
    dv_coefficients = defaultdict(float)
    
    if feasibility_flag:
        constant = _worker_model.objVal
        dual_values = _worker_model.getAttr('Pi', all_constrs)
        
        for constr_idx, row_entries in constr_nonant_map.items():
            pi = dual_values[constr_idx]
            if abs(pi) < 1e-12:
                continue
            
            for var_idx, coeff in row_entries:
                vname = nonant_idx_to_name[var_idx]
                dv_coef = -coeff * pi
                dv_coefficients[vname] += dv_coef
                constant -= dv_coef * nonant_values[var_idx]
    else:
        all_rhs = _worker_model._all_rhs
        farkas_values = _worker_model.getAttr('FarkasDual', all_constrs)
        constant = sum(pi * rhs for pi, rhs in zip(farkas_values, all_rhs))
        
        for constr_idx, row_entries in constr_nonant_map.items():
            pi = farkas_values[constr_idx]
            if abs(pi) < 1e-12:
                continue
            
            for var_idx, coeff in row_entries:
                vname = nonant_idx_to_name[var_idx]
                dv_coefficients[vname] -= coeff * pi
    
#        norm_factor = abs(constant)
#        if norm_factor > 1e+6:
#            constant /= norm_factor
#            for key in dv_coefficients:
#                dv_coefficients[key] /= norm_factor
    
    return _worker_model.objVal, constant, dict(dv_coefficients), feasibility_flag, status

def add_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_path_probabilities, master_var_cache):
    all_feasible = all(subproblem_feasibility.values())

    if all_feasible:
        constant_term = sum(subproblem_constants[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_probabilities)
        cut_expr = master_var_cache["theta"] - constant_term - quicksum(dv_coef * scenario_path_probabilities[sp_id] * master_var_cache[dv_name] for sp_id, dict_of_dvs in subproblem_dv_coefficients.items() for dv_name, dv_coef in dict_of_dvs.items())
        return cut_expr
    else:
        cut_exprs = []
        for sp_id, is_feasible in subproblem_feasibility.items():
            if not is_feasible:
                dv_dict = subproblem_dv_coefficients[sp_id]
                cut_expr = subproblem_constants[sp_id] + quicksum(dv_coef * master_var_cache[dv_name] for dv_name, dv_coef in dv_dict.items())
                cut_exprs.append(cut_expr)
        return cut_exprs

def add_multiple_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_paths, master_var_cache):
    cut_exprs = {}
    all_feasible = all(subproblem_feasibility.values())

    if all_feasible:
        for sp_id in scenario_paths:
            theta_var = master_var_cache[f"theta[{sp_id}]"]
            constant = subproblem_constants[sp_id]
            dv_dict = subproblem_dv_coefficients[sp_id]
            cut_exprs[sp_id] = theta_var - constant - quicksum(dv_coef * master_var_cache[dv_name] for dv_name, dv_coef in dv_dict.items())
    else:
        for sp_id, sub_feas in subproblem_feasibility.items():
            if not sub_feas:
                constant = subproblem_constants[sp_id]
                dv_dict = subproblem_dv_coefficients[sp_id]
                cut_exprs[sp_id] = constant + quicksum(dv_coef * master_var_cache[dv_name] for dv_name, dv_coef in dv_dict.items())
    
    return cut_exprs

def write_cuts(cuts_file, iteration, subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_path_probabilities, multi_cut_flag):
    lines = ['-' * 30, f"Iteration {iteration}:"]
    
    all_feasible = all(subproblem_feasibility.values())

    if multi_cut_flag:
        if all_feasible:
            for sp_id in scenario_path_probabilities.keys():
                parts = [f"theta[{sp_id}] >= {subproblem_constants[sp_id]:.3f}"]
                
                for dv_name, dv_coef in subproblem_dv_coefficients[sp_id].items():
                    if abs(dv_coef) > 1e-6:
                        sign = '+' if dv_coef >= 0 else '-'
                        parts.append(f" {sign} {abs(dv_coef):.3f} * {dv_name}")
                lines.append(''.join(parts))
        else:
            for sp_id, is_feasible in subproblem_feasibility.items():
                if not is_feasible:
                    parts = [f"0 <= {subproblem_constants[sp_id]:.3f}"]
                    for dv_name, dv_coef in subproblem_dv_coefficients[sp_id].items():
                        if abs(dv_coef) > 1e-6:
                            sign = '+' if dv_coef >= 0 else '-'
                            parts.append(f" {sign} {abs(dv_coef):.3f} * {dv_name}")
                    lines.append(''.join(parts))
    else:
        if all_feasible:
            constant_term = sum(subproblem_constants[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_probabilities.keys())
            parts = [f"theta >= {constant_term:.3f}"]
            
            aggregated_coeffs = {}
            for sp_id, dv_dict in subproblem_dv_coefficients.items():
                sp_prob = scenario_path_probabilities[sp_id]
                for dv_name, dv_coef in dv_dict.items():
                    aggregated_coeffs[dv_name] = aggregated_coeffs.get(dv_name, 0.0) + dv_coef * sp_prob
            
            for dv_name, coef in aggregated_coeffs.items():
                if abs(coef) > 1e-6:
                    sign = '+' if coef >= 0 else '-'
                    parts.append(f" {sign} {abs(coef):.3f} * {dv_name}")
            lines.append(''.join(parts))
        else:
            for sp_id, is_feasible in subproblem_feasibility.items():
                if not is_feasible:
                    parts = [f"0 <= {subproblem_constants[sp_id]:.3f}"]
                    for dv_name, dv_coef in subproblem_dv_coefficients[sp_id].items():
                        if abs(dv_coef) > 1e-6:
                            sign = '+' if dv_coef >= 0 else '-'
                            parts.append(f" {sign} {abs(dv_coef):.3f} * {dv_name}")
                    lines.append(f"SP{sp_id}: " + ''.join(parts))
    
    cuts_file.write('\n'.join(lines) + '\n')
    cuts_file.flush()

def get_leaf_node_solution(leaf_node_id, leaf_parent_node_id, numStages, numSubperiods, numSubterms):
    _worker_model = _cached_worker_model
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
        
        with call_back_data['lock']:
            call_back_data['iteration'] += 1

        nonant_values = model.cbGetSolution(call_back_data['nonant_vars'])
        nonanticipativity_lookup = dict(zip(call_back_data['nonant_var_names'], nonant_values))

        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if call_back_data['continuous_flag']:
            lower_bound = current_obj
        else:
            lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        subproblem_start_time = time.time()
        futures = {sp_id: call_back_data['executors'][sp_id].submit(solve_subproblem, nonanticipativity_lookup) for sp_id in call_back_data['scenario_paths'].keys()}
        subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
        subproblem_execution_time = time.time() - subproblem_start_time

        subproblem_objectives = {sp_id: result[0] for sp_id, result in subproblem_results.items()}
        subproblem_constants = {sp_id: result[1] for sp_id, result in subproblem_results.items()}
        subproblem_dv_coefficients = {sp_id: result[2] for sp_id, result in subproblem_results.items()}
        subproblem_feasibility = {sp_id: result[3] for sp_id, result in subproblem_results.items()}
        subproblem_statuses = {sp_id: result[4] for sp_id, result in subproblem_results.items()}
        
        unexpected_statuses = [(sp_id, status) for sp_id, status in subproblem_statuses.items() if status != GRB.OPTIMAL and status != GRB.INFEASIBLE]
        if unexpected_statuses:
            status_names = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
            with open(os.path.join(call_back_data['results_directory'], 'SubproblemStatusLog.txt'), 'a') as status_file:
                for sp_id, status in unexpected_statuses:
                    status_name = status_names.get(status, f'UNKNOWN({status})')
                    status_file.write(f"Iteration {call_back_data['iteration']}: Subproblem {sp_id} status: {status_name}\n")
        
        all_feasible = all(subproblem_feasibility.values())
        
        if call_back_data['multi_cut_flag']:
            if 'theta_vars' not in call_back_data:
                call_back_data['theta_vars'] = {sp_id: call_back_data['master_var_cache'][f"theta[{sp_id}]"] for sp_id in call_back_data['scenario_paths'].keys()}
                call_back_data['theta_vars_list'] = list(call_back_data['theta_vars'].values())
            theta_values = model.cbGetSolution(call_back_data['theta_vars_list'])
            scenario_path_probabilities = call_back_data['scenario_path_probabilities']
            scenario_path_keys = list(call_back_data['scenario_paths'].keys())
            theta_sum = sum(tv * scenario_path_probabilities[sp_id] for tv, sp_id in zip(theta_values, scenario_path_keys))
            subproblem_obj_sum = sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_keys)
            upper_bound = current_obj - theta_sum + subproblem_obj_sum
            cut_expressions = add_multiple_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, call_back_data['scenario_paths'], call_back_data['master_var_cache'])
            for cut_expression in cut_expressions.values():
                model.cbLazy(cut_expression >= 0)
        else:
            if 'theta_var' not in call_back_data:
                call_back_data['theta_var'] = call_back_data['master_var_cache']["theta"]
            scenario_path_probabilities = call_back_data['scenario_path_probabilities']
            scenario_path_keys = list(call_back_data['scenario_paths'].keys())
            subproblem_obj_sum = sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_keys)
            upper_bound = current_obj - model.cbGetSolution(call_back_data['theta_var']) + subproblem_obj_sum
            cut_result = add_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_path_probabilities, call_back_data['master_var_cache'])
            if isinstance(cut_result, list):
                for cut_expression in cut_result:
                    model.cbLazy(cut_expression >= 0)
            else:
                model.cbLazy(cut_result >= 0)

        with call_back_data['lock']:
            if call_back_data['cuts_file']:
                write_cuts(call_back_data['cuts_file'], call_back_data['iteration'], subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, call_back_data['scenario_path_probabilities'], call_back_data['multi_cut_flag'])

            if all_feasible and upper_bound < call_back_data['best_upper_bound']:
                call_back_data['best_upper_bound'] = upper_bound
                all_vars = model.getVars()
                all_var_values = model.cbGetSolution(all_vars)
                call_back_data['best_ub_lookup'] = {var.varName: val for var, val in zip(all_vars, all_var_values) if not var.varName.startswith("theta")}

            if lower_bound > call_back_data['best_lower_bound']:
                call_back_data['best_lower_bound'] = lower_bound
            
            gap = (call_back_data['best_upper_bound'] - call_back_data['best_lower_bound']) / max(1e-6, call_back_data['best_upper_bound'])
            
            log_lines = [
                '-' * 30,
                f"Iteration {call_back_data['iteration']}:",
                f"Upper Bound: {call_back_data['best_upper_bound']:.2f}",
                f"Lower Bound: {call_back_data['best_lower_bound']:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Iteration Time: {time.time() - iteration_start_time:.2f} seconds",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds"
            ]
            call_back_data['log_file'].write('\n'.join(log_lines) + '\n')
            call_back_data['log_file'].flush()

            if all_feasible and gap < call_back_data['tolerance']:
                model.terminate()

def CampusApplication(numStages, numSubperiods, numSubterms, scenarioTree, initial_tech, emission_limits, electricity_demand, heat_demand, 
                      budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, discount_factor, scenario_paths, 
                      scenario_path_probabilities, tolerance, benders_without_feasibility_flag, multi_cut_flag, callback_flag, write_cuts_flag, 
                      continuous_flag, feasibility_cut_intervals, master_threads, threads_per_worker, incumbent_solution):
    
    if benders_without_feasibility_flag:
        from benders_model_feas import MasterProblemModel, SubProblemModel, OperationalNonanticipativityModel
    else:
        from benders_model import MasterProblemModel, SubProblemModel, OperationalNonanticipativityModel

    execution_start_time = time.time()

    executors = {}
    for scenario_path_id, scenario_path_nodes in scenario_paths.items():
        scenarioTree_copy = copy.deepcopy(scenarioTree)
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_worker_subproblem,
            initargs=(SubProblemModel, scenario_path_id, scenario_path_nodes, scenarioTree_copy, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads_per_worker, discount_factor)
        )
        executors[scenario_path_id] = executor

    master_model = MasterProblemModel(copy.deepcopy(scenarioTree), emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor, multi_cut_flag, scenario_paths, scenario_path_probabilities, continuous_flag, feasibility_cut_intervals, tolerance)

    log_file = open(os.path.join(results_directory, 'BendersLog.txt'), 'w')
    cuts_file = open(os.path.join(results_directory, 'GeneratedCuts.txt'), 'w') if write_cuts_flag else None

    nonant_vars = [var for var in master_model.getVars() if not var.varName.startswith("theta")]
    nonant_var_names = [var.varName for var in nonant_vars]
    master_var_cache = {var.varName: var for var in master_model.getVars()}

    if incumbent_solution is not None:
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, incumbent_solution) for sp_id in scenario_paths.keys()}
        
        incumbent_sp_results = {sp_id: future.result() for sp_id, future in futures.items()}
        incumbent_sp_constants = {sp_id: result[1] for sp_id, result in incumbent_sp_results.items()}
        incumbent_sp_dv_coefficients = {sp_id: result[2] for sp_id, result in incumbent_sp_results.items()}
        incumbent_sp_feasibility = {sp_id: result[3] for sp_id, result in incumbent_sp_results.items()}
        
        all_incumbent_feasible = all(incumbent_sp_feasibility.values())
        
        if all_incumbent_feasible:            
            if multi_cut_flag:
                cut_exprs = add_multiple_cuts(incumbent_sp_constants, incumbent_sp_dv_coefficients, incumbent_sp_feasibility, scenario_paths, master_var_cache)
                for sp_id, cut_expr in cut_exprs.items():
                    master_model.addConstr(cut_expr >= 0, name=f"incumbent_opt_cut_{sp_id}")
            else:
                cut_expr = add_cuts(incumbent_sp_constants, incumbent_sp_dv_coefficients, incumbent_sp_feasibility, scenario_path_probabilities, master_var_cache)
                master_model.addConstr(cut_expr >= 0, name="incumbent_opt_cut")
            
            master_model.update()
            
            log_file.write(f"Incumbent solution is feasible\n")
            
            if write_cuts_flag:
                write_cuts(cuts_file, 0, incumbent_sp_constants, incumbent_sp_dv_coefficients, incumbent_sp_feasibility, scenario_path_probabilities, multi_cut_flag)

        log_file.write("-" * 30 + "\n")
        log_file.flush()

    if callback_flag:
        master_model.setParam('LazyConstraints', 1)
        nonant_var_names = [var.varName for var in nonant_vars]

        master_model._callback_data = {
            'iteration': 0,
            'lock': threading.Lock(),
            'log_file': log_file,
            'cuts_file': cuts_file,
            'executors': executors,
            'scenario_paths': scenario_paths,
            'scenario_path_probabilities': scenario_path_probabilities,
            'multi_cut_flag' : multi_cut_flag,
            'best_upper_bound': float('inf'),
            'best_lower_bound': float('-inf'),
            'best_ub_lookup': None,
            'nonant_vars': nonant_vars,
            'nonant_var_names': nonant_var_names,
            'master_var_cache': master_var_cache,
            'tolerance': tolerance,
            'results_directory': results_directory,
            'continuous_flag': continuous_flag
        }

        master_model.optimize(benders_callback)
        iteration = master_model._callback_data['iteration']
        best_upper_bound = master_model._callback_data['best_upper_bound']
        lower_bound = master_model._callback_data['best_lower_bound']
        best_ub_lookup = master_model._callback_data['best_ub_lookup']

    else:
        iteration = 0
        max_iterations = 10000
        best_upper_bound = float('inf')
        best_lower_bound = float('-inf')
        best_ub_lookup = None
        previous_cut_data = None

        while iteration < max_iterations:
            iteration += 1
            master_start_time = time.time()
            master_model.optimize()
            master_execution_time = time.time() - master_start_time
            
            if continuous_flag:
                lower_bound = master_model.ObjVal
            else:
                lower_bound = master_model.ObjBound
            
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound

            nonant_solution_values = master_model.getAttr('X', nonant_vars)
            nonanticipativity_lookup = dict(zip(nonant_var_names, nonant_solution_values))

            subproblem_start_time = time.time()
            futures = {sp_id: executors[sp_id].submit(solve_subproblem, nonanticipativity_lookup) for sp_id in scenario_paths.keys()}
            subproblem_results = {sp_id: futures[sp_id].result() for sp_id in futures.keys()}
            subproblem_execution_time = time.time() - subproblem_start_time

            subproblem_objectives = {sp_id: result[0] for sp_id, result in subproblem_results.items()}
            subproblem_constants = {sp_id: result[1] for sp_id, result in subproblem_results.items()}
            subproblem_dv_coefficients = {sp_id: result[2] for sp_id, result in subproblem_results.items()}
            subproblem_feasibility = {sp_id: result[3] for sp_id, result in subproblem_results.items()}
            subproblem_statuses = {sp_id: result[4] for sp_id, result in subproblem_results.items()}
            
            unexpected_statuses = [(sp_id, status) for sp_id, status in subproblem_statuses.items() if status != GRB.OPTIMAL and status != GRB.INFEASIBLE]
            if unexpected_statuses:
                status_names = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
                with open(os.path.join(results_directory, 'SubproblemStatusLog.txt'), 'a') as status_file:
                    for sp_id, status in unexpected_statuses:
                        status_name = status_names.get(status, f'UNKNOWN({status})')
                        status_file.write(f"Iteration {iteration}: Subproblem {sp_id} status: {status_name}\n")

            all_feasible = all(subproblem_feasibility.values())
            
            if multi_cut_flag:
                upper_bound = master_model.ObjVal - sum([master_var_cache[f"theta[{sp_id}]"].X * sp_prob for sp_id, sp_prob in scenario_path_probabilities.items()]) + sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_paths.keys())
                cut_expressions = add_multiple_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_paths, master_var_cache)
                for sp_id, cut_expression in cut_expressions.items():
                    cut_name = f'OptimalityCut{sp_id}_{iteration}' if all_feasible else f'FeasibilityCut{sp_id}_{iteration}'
                    master_model.addConstr(cut_expression >= 0, name=cut_name)
            else:
                upper_bound = master_model.ObjVal - master_var_cache["theta"].X + sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_paths.keys())
                cut_result = add_cuts(subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_path_probabilities, master_var_cache)
                
                if isinstance(cut_result, list):
                    for idx, cut_expression in enumerate(cut_result):
                        master_model.addConstr(cut_expression >= 0, name=f'FeasibilityCut_{iteration}_{idx}')
                else:
                    master_model.addConstr(cut_result >= 0, name=f'OptimalityCut_{iteration}')

            if cuts_file:
                write_cuts(cuts_file, iteration, subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility, scenario_path_probabilities, multi_cut_flag)

            current_cut_data = (subproblem_constants, subproblem_dv_coefficients, subproblem_feasibility)
            if previous_cut_data is not None and not continuous_flag:
                prev_constants, prev_dv_coefficients, prev_feasibility = previous_cut_data
                if (subproblem_constants == prev_constants and 
                    subproblem_dv_coefficients == prev_dv_coefficients and 
                    subproblem_feasibility == prev_feasibility):
                    current_mipgap = master_model.Params.MIPGap
                    new_mipgap = float(current_mipgap) * 0.5
                    master_model.setParam('MIPGap', new_mipgap)
            previous_cut_data = current_cut_data

            if all_feasible and upper_bound < best_upper_bound:
                best_upper_bound = upper_bound
                best_ub_vars = master_model.getVars()
                best_ub_var_values = master_model.getAttr('X', best_ub_vars)
                best_ub_lookup = {var.varName: val for var, val in zip(best_ub_vars, best_ub_var_values) if not var.varName.startswith("theta")}
    
            gap = (best_upper_bound - best_lower_bound) / max(1e-6, best_upper_bound)
            log_lines = [
                '-' * 30,
                f"Iteration {iteration}:",
                f"Upper Bound: {best_upper_bound:.2f}",
                f"Lower Bound: {best_lower_bound:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Master Problem Execution Time: {master_execution_time:.2f} seconds",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds"
            ]
            log_file.write('\n'.join(log_lines) + '\n')
            log_file.flush()

            if all_feasible and gap < tolerance:
                break

    final_gap = (best_upper_bound - best_lower_bound) / max(1e-6, best_upper_bound)
    summary_lines = [
        "=" * 30,
        "Final Summary",
        f"Total Iterations: {iteration}",
        f"Best Upper Bound: {best_upper_bound:.2f}",
        f"Final Lower Bound: {best_lower_bound:.2f}",
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
        
        e_discharge = max(0, e_discharge_eff * (e_carry_prev - e_carry_curr))
        h_discharge = max(0, h_discharge_eff * (h_carry_prev - h_carry_curr))
        
        discharge_lines.append(f"electricitydischarge_{leaf_node_id}[{last_period + 1},1] {e_discharge}\n")
        discharge_lines.append(f"heatdischarge_{leaf_node_id}[{last_period + 1},1] {h_discharge}\n")
    
    with open(final_sol_file, 'a') as f:
        f.writelines(discharge_lines)