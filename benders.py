from gurobipy import GRB, LinExpr
import concurrent.futures
import numpy as np
import threading
import numba
import time
import os

_cached_worker_model = None

def _init_worker_subproblem(subproblem_builder, *args):
    global _cached_worker_model
    _cached_worker_model = subproblem_builder(*args)

def get_nonant_var_names():
    return _cached_worker_model._nonant_var_names

def solve_subproblem(nonant_bounds):
    _worker_model = _cached_worker_model

    _worker_model.setAttr('LB', _worker_model._nonant_vars, nonant_bounds)
    _worker_model.setAttr('UB', _worker_model._nonant_vars, nonant_bounds)

    _worker_model.optimize()

    status = _worker_model.status
    is_optimal = status == GRB.OPTIMAL

    if is_optimal:
        objective_value = _worker_model.objVal
        dual_values = np.array(_worker_model.getAttr('Pi', _worker_model._all_constrs))
        dv_array = np.asarray(dual_values @ _worker_model._A_nonant)
        constant = objective_value + np.dot(dv_array, nonant_bounds)
        np.negative(dv_array, out=dv_array)
    else:
        objective_value = float('inf')
        farkas_values = np.array(_worker_model.getAttr('FarkasDual', _worker_model._all_constrs))
        constant = np.dot(farkas_values, _worker_model._all_rhs)
        dv_array = np.asarray(farkas_values @ _worker_model._A_nonant)
        np.negative(dv_array, out=dv_array)

    return objective_value, constant, dv_array, is_optimal, status

def _compute_scale_factor(constant, dv_array, significant_idx):
    max_abs_coef = np.max(np.abs(dv_array[significant_idx])) if len(significant_idx) > 0 else 0.0
    max_abs_value = max(abs(constant), max_abs_coef)
    return 1.0 / (max_abs_value / 1e+06) if max_abs_value > 1e+06 else 1.0

def _build_cut_linexpr(dv_array, significant_idx, scale_factor, master_vars, constant, theta_var=None):
    expr = LinExpr()
    if theta_var is not None:
        if len(significant_idx) > 0:
            coefficients = (dv_array[significant_idx] * (-scale_factor)).tolist()
            coefficients.append(scale_factor)
            vars_list = [master_vars[i] for i in significant_idx]
            vars_list.append(theta_var)
            expr.addTerms(coefficients, vars_list)
        else:
            expr.addTerms([scale_factor], [theta_var])
        expr.addConstant(-constant * scale_factor)
    else:
        if len(significant_idx) > 0:
            coefficients = (dv_array[significant_idx] * scale_factor).tolist()
            vars_list = [master_vars[i] for i in significant_idx]
            expr.addTerms(coefficients, vars_list)
        expr.addConstant(constant * scale_factor)
    return expr

def add_benders_cuts(subproblem_constants, subproblem_dv_arrays, subproblem_feasibility, scenario_paths, master_nonant_vars_by_sp, theta_vars):
    cut_expressions = {}
    all_feasible = all(subproblem_feasibility.values())

    if all_feasible:
        for sp_id in scenario_paths:
            dv_array = subproblem_dv_arrays[sp_id]
            constant = subproblem_constants[sp_id]
            significant_idx = np.where(np.abs(dv_array) > 1e-6)[0]
            scale_factor = _compute_scale_factor(constant, dv_array, significant_idx)
            cut_expressions[sp_id] = _build_cut_linexpr(dv_array, significant_idx, scale_factor, master_nonant_vars_by_sp[sp_id], constant, theta_vars[sp_id])
    else:
        for sp_id, is_feasible in subproblem_feasibility.items():
            if not is_feasible:
                constant = subproblem_constants[sp_id]
                dv_array = subproblem_dv_arrays[sp_id]
                significant_idx = np.where(np.abs(dv_array) > 1e-6)[0]
                scale_factor = _compute_scale_factor(constant, dv_array, significant_idx)
                cut_expressions[sp_id] = _build_cut_linexpr(dv_array, significant_idx, scale_factor, master_nonant_vars_by_sp[sp_id], constant)

    return cut_expressions

@numba.njit(cache=True)
def minimum_sum_contiguous_subarray(array):
    n = len(array)
    
    min_ending_here = array[0]
    min_so_far = array[0]
    
    current_start = 0
    best_start = 0
    best_end = 0
    
    for i in range(1, n):
        if array[i] < min_ending_here + array[i]:
            min_ending_here = array[i]
            current_start = i
        else:
            min_ending_here += array[i]
        
        if min_ending_here < min_so_far:
            min_so_far = min_ending_here
            best_start = current_start
            best_end = i
    
    return min_so_far, best_start + 1, best_end + 1

def _build_electricity_cut(sp_id, sp_separation_data, electricity_storage_const, q_lb_e, q_ub_e, min_sum_e):
    if min_sum_e + electricity_storage_const >= 0:
        return None, None

    electricity_demand_sum = sp_separation_data['electricity_demand_cumsum'][q_ub_e] - sp_separation_data['electricity_demand_cumsum'][q_lb_e - 1]
    scale_factor = 1.0 / (electricity_demand_sum / 1e+06) if electricity_demand_sum > 1e+06 else 1.0
    summed_coefs = sp_separation_data['elec_coef_cumsum'][q_ub_e] - sp_separation_data['elec_coef_cumsum'][q_lb_e - 1]

    cut_name = f'ValidInequality_Electricity_SP{sp_id}_q{q_lb_e}_{q_ub_e}'
    expr = LinExpr()
    expr.addTerms((summed_coefs * scale_factor).tolist(), sp_separation_data['elec_gen_vars'])
    expr.addTerms((sp_separation_data['elec_storage_coefs'] * scale_factor).tolist(), sp_separation_data['elec_storage_vars'])
    expr.addConstant(-electricity_demand_sum * scale_factor)

    return cut_name, expr

def _build_heat_cut(sp_id, sp_separation_data, heat_storage_const, q_lb_h, q_ub_h, min_sum_h):
    if min_sum_h + heat_storage_const >= 0:
        return None, None

    heat_demand_sum = sp_separation_data['heat_demand_cumsum'][q_ub_h] - sp_separation_data['heat_demand_cumsum'][q_lb_h - 1]
    scale_factor = 1.0 / (heat_demand_sum / 1e+06) if heat_demand_sum > 1e+06 else 1.0
    num_subterms = q_ub_h - q_lb_h + 1
    summed_coefs = sp_separation_data['heat_coef_cumsum'][q_ub_h] - sp_separation_data['heat_coef_cumsum'][q_lb_h - 1]

    cut_name = f'ValidIneq_Heat_SP{sp_id}_q{q_lb_h}_{q_ub_h}'
    expr = LinExpr()
    expr.addTerms((summed_coefs * scale_factor).tolist(), sp_separation_data['heat_gen_vars'])
    expr.addTerms((sp_separation_data['heat_transfer_coefs'] * scale_factor * num_subterms).tolist(), sp_separation_data['heat_transfer_vars'])
    expr.addTerms((sp_separation_data['heat_storage_coefs'] * scale_factor).tolist(), sp_separation_data['heat_storage_vars'])
    expr.addConstant(-heat_demand_sum * scale_factor)

    return cut_name, expr

def add_valid_inequalities(separation_data, subproblem_feasibility=None, callback_flag=False, master_model=None, initial_iteration=False, numSubterms=None, scenario_paths=None):
    cut_expressions = {}

    if initial_iteration:
        for sp_id in scenario_paths.keys():
            sp_separation_data = separation_data[sp_id]

            electricity_demand_sum = sp_separation_data['electricity_demand_cumsum'][numSubterms]
            scale_factor = 1.0 / (electricity_demand_sum / 1e+06) if electricity_demand_sum > 1e+06 else 1.0
            elec_summed_coefs = sp_separation_data['elec_coef_cumsum'][numSubterms]
            expr_e = LinExpr()
            expr_e.addTerms((elec_summed_coefs * scale_factor).tolist(), sp_separation_data['elec_gen_vars'])
            expr_e.addTerms((sp_separation_data['elec_storage_coefs'] * scale_factor).tolist(), sp_separation_data['elec_storage_vars'])
            expr_e.addConstant(-electricity_demand_sum * scale_factor)
            cut_expressions[f'ValidInequality_Electricity_SP{sp_id}_q{1}_{numSubterms}'] = expr_e

            heat_demand_sum = sp_separation_data['heat_demand_cumsum'][numSubterms]
            scale_factor = 1.0 / (heat_demand_sum / 1e+06) if heat_demand_sum > 1e+06 else 1.0
            heat_summed_coefs = sp_separation_data['heat_coef_cumsum'][numSubterms]
            expr_h = LinExpr()
            expr_h.addTerms((heat_summed_coefs * scale_factor).tolist(), sp_separation_data['heat_gen_vars'])
            expr_h.addTerms((sp_separation_data['heat_transfer_coefs'] * scale_factor * numSubterms).tolist(), sp_separation_data['heat_transfer_vars'])
            expr_h.addTerms((sp_separation_data['heat_storage_coefs'] * scale_factor).tolist(), sp_separation_data['heat_storage_vars'])
            expr_h.addConstant(-heat_demand_sum * scale_factor)
            cut_expressions[f'ValidIneq_Heat_SP{sp_id}_q{1}_{numSubterms}'] = expr_h

        return cut_expressions

    for sp_id, is_feasible in subproblem_feasibility.items():
        if is_feasible:
            continue

        sp_separation_data = separation_data[sp_id]
        electricity_demand = sp_separation_data['electricity_demand']
        heat_demand = sp_separation_data['heat_demand']
        elec_gen_coef_matrix = sp_separation_data['elec_gen_coef_matrix']
        heat_gen_coef_matrix = sp_separation_data['heat_gen_coef_matrix']

        if callback_flag:
            elec_gen_vals = np.array(master_model.cbGetSolution(sp_separation_data['elec_gen_vars']))
            heat_gen_vals = np.array(master_model.cbGetSolution(sp_separation_data['heat_gen_vars']))
            elec_storage_vals = np.array(master_model.cbGetSolution(sp_separation_data['elec_storage_vars']))
            heat_transfer_vals = np.array(master_model.cbGetSolution(sp_separation_data['heat_transfer_vars']))
            heat_storage_vals = np.array(master_model.cbGetSolution(sp_separation_data['heat_storage_vars']))
        else:
            elec_gen_vals = np.array(master_model.getAttr('X', sp_separation_data['elec_gen_vars']))
            heat_gen_vals = np.array(master_model.getAttr('X', sp_separation_data['heat_gen_vars']))
            elec_storage_vals = np.array(master_model.getAttr('X', sp_separation_data['elec_storage_vars']))
            heat_transfer_vals = np.array(master_model.getAttr('X', sp_separation_data['heat_transfer_vars']))
            heat_storage_vals = np.array(master_model.getAttr('X', sp_separation_data['heat_storage_vars']))

        electricity_storage_const = np.dot(sp_separation_data['elec_storage_coefs'], elec_storage_vals)
        heat_transfer_per_subperiod = np.dot(sp_separation_data['heat_transfer_coefs'], heat_transfer_vals)
        heat_storage_const = np.dot(sp_separation_data['heat_storage_coefs'], heat_storage_vals)

        electricity_contiguous_array = elec_gen_coef_matrix @ elec_gen_vals - electricity_demand
        heat_contiguous_array = (heat_gen_coef_matrix @ heat_gen_vals) + heat_transfer_per_subperiod - heat_demand

        min_sum_e, q_lb_e, q_ub_e = minimum_sum_contiguous_subarray(np.ascontiguousarray(electricity_contiguous_array))
        min_sum_h, q_lb_h, q_ub_h = minimum_sum_contiguous_subarray(np.ascontiguousarray(heat_contiguous_array))

        elec_cut_name, elec_cut_expr = _build_electricity_cut(sp_id, sp_separation_data, electricity_storage_const, q_lb_e, q_ub_e, min_sum_e)
        heat_cut_name, heat_cut_expr = _build_heat_cut(sp_id, sp_separation_data, heat_storage_const, q_lb_h, q_ub_h, min_sum_h)

        if elec_cut_name is not None:
            cut_expressions[elec_cut_name] = elec_cut_expr
        if heat_cut_name is not None:
            cut_expressions[heat_cut_name] = heat_cut_expr

    return cut_expressions

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

def get_all_subproblem_solution():
    _worker_model = _cached_worker_model
    all_vars = {}
    for var in _worker_model.getVars():
        if not var.varName.startswith('plus_'):
            all_vars[var.varName] = var.X
    return all_vars

def write_final_subproblem_solutions(executors, nonant_solution_array, master_dv_indices, results_directory, scenario_paths, numStages, numSubperiods, numSubterms):
    futures = {sp_id: executors[sp_id].submit(solve_subproblem, nonant_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}
    
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
        nonant_solution_array = np.array(nonant_values, dtype=np.float64)
        master_dv_indices = call_back_data['master_dv_indices']

        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if call_back_data['continuous_flag']:
            lower_bound = current_obj
        else:
            lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        subproblem_start_time = time.time()
        futures = {sp_id: call_back_data['executors'][sp_id].submit(solve_subproblem, nonant_solution_array[master_dv_indices[sp_id]]) for sp_id in call_back_data['scenario_paths'].keys()}
        subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
        subproblem_execution_time = time.time() - subproblem_start_time

        subproblem_objectives = {}
        subproblem_constants = {}
        subproblem_dv_arrays = {}
        subproblem_feasibility = {}
        subproblem_statuses = {}
        for sp_id, result in subproblem_results.items():
            subproblem_objectives[sp_id] = result[0]
            subproblem_constants[sp_id] = result[1]
            subproblem_dv_arrays[sp_id] = result[2]
            subproblem_feasibility[sp_id] = result[3]
            subproblem_statuses[sp_id] = result[4]

        unexpected_statuses = [(sp_id, status) for sp_id, status in subproblem_statuses.items() if status != GRB.OPTIMAL and status != GRB.INFEASIBLE]
        if unexpected_statuses:
            with open(os.path.join(call_back_data['results_directory'], 'SubproblemStatusLog.txt'), 'a') as status_file:
                for sp_id, status in unexpected_statuses:
                    status_file.write(f"Iteration {call_back_data['iteration']}: Subproblem {sp_id} status: {status}\n")

        all_feasible = all(subproblem_feasibility.values())

        if 'theta_vars_list' not in call_back_data:
            call_back_data['theta_vars_list'] = list(call_back_data['theta_vars'].values())
        theta_values = model.cbGetSolution(call_back_data['theta_vars_list'])
        scenario_path_probabilities = call_back_data['scenario_path_probabilities']
        scenario_path_keys = call_back_data['scenario_path_keys']
        theta_sum = sum(theta_value * scenario_path_probabilities[sp_id] for theta_value, sp_id in zip(theta_values, scenario_path_keys))
        subproblem_obj_sum = sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_keys)
        upper_bound = current_obj - theta_sum + subproblem_obj_sum
        cut_expressions = add_benders_cuts(subproblem_constants, subproblem_dv_arrays, subproblem_feasibility, call_back_data['scenario_paths'], call_back_data['master_nonant_vars_by_sp'], call_back_data['theta_vars'])
        for cut_expression in cut_expressions.values():
            model.cbLazy(cut_expression >= 0)

        valid_inequality_derivation_time = 0
        num_valid_inequalities = 0
        if not all_feasible and call_back_data['valid_inequalities_flag']:
            valid_inequality_start_time = time.time()
            valid_ineq_cut_expressions = add_valid_inequalities(call_back_data['separation_data'], subproblem_feasibility=subproblem_feasibility, callback_flag=True, master_model=model)
            num_valid_inequalities = len(valid_ineq_cut_expressions)
            for _, cut_expression in valid_ineq_cut_expressions.items():
                model.cbLazy(cut_expression >= 0)
            valid_inequality_derivation_time = time.time() - valid_inequality_start_time

        with call_back_data['lock']:
            call_back_data['total_subproblem_time'] += subproblem_execution_time

            if all_feasible:
                call_back_data['optimality_cut_iterations'] += 1
            else:
                call_back_data['feasibility_cut_iterations'] += 1

            if valid_inequality_derivation_time > 0:
                call_back_data['total_valid_inequality_time'] += valid_inequality_derivation_time
                call_back_data['valid_inequalities_added'] += num_valid_inequalities

            if all_feasible and upper_bound < call_back_data['best_upper_bound']:
                call_back_data['best_upper_bound'] = upper_bound
                all_vars = model.getVars()
                all_var_values = model.cbGetSolution(all_vars)
                call_back_data['best_ub_solution'] = {var.varName: val for var, val in zip(all_vars, all_var_values) if not var.varName.startswith("theta")}

            if lower_bound > call_back_data['best_lower_bound']:
                call_back_data['best_lower_bound'] = lower_bound
            
            gap = (call_back_data['best_upper_bound'] - call_back_data['best_lower_bound']) / max(1e-6, call_back_data['best_upper_bound'])
            
            log_lines = [
                '-' * 30,
                f"Iteration {call_back_data['iteration']}:",
                f"Upper Bound: {call_back_data['best_upper_bound']:.2f}",
                f"Lower Bound: {call_back_data['best_lower_bound']:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds",
                f"Iteration Time: {time.time() - iteration_start_time:.2f} seconds"
            ]
            
            if valid_inequality_derivation_time > 0:
                log_lines.append(f"Valid Inequality Derivation Time: {valid_inequality_derivation_time:.2f} seconds")
            
            call_back_data['log_file'].write('\n'.join(log_lines) + '\n')
            call_back_data['total_iteration_time'] += time.time() - iteration_start_time

            if gap < call_back_data['tolerance']:
                model.terminate()

def CampusApplication(numStages, numSubperiods, numSubterms, scenarioTree, initial_tech, emission_limits, electricity_demand, 
                      heat_demand, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, log_file, 
                      discount_factor, scenario_paths, scenario_path_probabilities, tolerance, benders_without_feasibility_flag, 
                      aggregated_subproblems_flag, callback_flag, continuous_flag, valid_inequalities_flag,
                      master_threads, threads_per_worker, incumbent_solution, incumbent_solve=False):
    
    if benders_without_feasibility_flag:
        from benders_model_feas import MasterProblemModel, SubProblemModel, OperationalNonanticipativityModel
    else:
        from benders_model import MasterProblemModel, SubProblemModel, OperationalNonanticipativityModel

    executors = {}
    for scenario_path_id, scenario_path_nodes in scenario_paths.items():
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_worker_subproblem,
            initargs=(SubProblemModel, scenario_path_id, scenario_path_nodes, scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, threads_per_worker, discount_factor, aggregated_subproblems_flag)
        )
        executors[scenario_path_id] = executor

    sp_nonant_names_futures = {sp_id: executors[sp_id].submit(get_nonant_var_names) for sp_id in scenario_paths}
    sp_nonant_names = {sp_id: future.result() for sp_id, future in sp_nonant_names_futures.items()}

    master_model, master_env, separation_data = MasterProblemModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor, scenario_paths, scenario_path_probabilities, continuous_flag, valid_inequalities_flag, tolerance)

    nonant_vars = [var for var in master_model.getVars() if not var.varName.startswith("theta")]
    nonant_var_names = [var.varName for var in nonant_vars]
    master_var_cache = {var.varName: var for var in master_model.getVars()}

    master_nonant_vars_by_sp = {}
    sp_to_master_idx = {}
    master_nonant_name_to_idx = {name: i for i, name in enumerate(nonant_var_names)}
    for sp_id in scenario_paths:
        sp_names = sp_nonant_names[sp_id]
        master_nonant_vars_by_sp[sp_id] = [master_var_cache[name] for name in sp_names]
        sp_to_master_idx[sp_id] = [master_nonant_name_to_idx[name] for name in sp_names]

    master_dv_indices = {sp_id: np.array(sp_to_master_idx[sp_id]) for sp_id in scenario_paths}

    theta_vars = {sp_id: master_var_cache[f"theta[{sp_id}]"] for sp_id in scenario_paths}
    theta_vars_values_list = list(theta_vars.values())

    if separation_data is not None:
        for path_id, scenario_path_data in separation_data.items():
            scenario_path_data['elec_gen_vars'] = [master_var_cache[var_name] for var_name in scenario_path_data['elec_gen_var_names']]
            scenario_path_data['heat_gen_vars'] = [master_var_cache[var_name] for var_name in scenario_path_data['heat_gen_var_names']]
            scenario_path_data['elec_storage_vars'] = [master_var_cache[var_name] for var_name in scenario_path_data['elec_storage_var_names']]
            scenario_path_data['heat_storage_vars'] = [master_var_cache[var_name] for var_name in scenario_path_data['heat_storage_var_names']]
            scenario_path_data['heat_transfer_vars'] = [master_var_cache[var_name] for var_name in scenario_path_data['heat_transfer_var_names']]
            scenario_path_data['elec_storage_coefs'] = np.array(scenario_path_data['elec_storage_coefs'], dtype=np.float64)
            scenario_path_data['heat_storage_coefs'] = np.array(scenario_path_data['heat_storage_coefs'], dtype=np.float64)
            scenario_path_data['heat_transfer_coefs'] = np.array(scenario_path_data['heat_transfer_coefs'], dtype=np.float64)

    generated_cuts = {}
    if valid_inequalities_flag:
        valid_ineq_cut_expressions = add_valid_inequalities(separation_data, initial_iteration=True, numSubterms=numSubterms, scenario_paths=scenario_paths)
        for cut_name, cut_expression in valid_ineq_cut_expressions.items():
            cut = master_model.addConstr(cut_expression >= 0, name=f'{cut_name}_{0}')
            generated_cuts[f'{cut_name}_{0}'] = {'constraint': cut, 'inactive_count': 0}

    total_master_time = 0
    total_iteration_time = 0 if callback_flag else time.time()
    total_subproblem_time = 0
    total_valid_inequality_time = 0

    feasibility_cut_iterations = 0
    optimality_cut_iterations = 0
    valid_inequalities_added = len(valid_ineq_cut_expressions) if valid_inequalities_flag else 0

    if incumbent_solution is not None:
        incumbent_nonant_array = np.array([incumbent_solution[name] for name in nonant_var_names], dtype=np.float64)
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, incumbent_nonant_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}

        incumbent_sp_results = {sp_id: future.result() for sp_id, future in futures.items()}
        incumbent_sp_constants = {sp_id: result[1] for sp_id, result in incumbent_sp_results.items()}
        incumbent_sp_dv_arrays = {sp_id: result[2] for sp_id, result in incumbent_sp_results.items()}
        incumbent_sp_feasibility = {sp_id: result[3] for sp_id, result in incumbent_sp_results.items()}

        all_incumbent_feasible = all(incumbent_sp_feasibility.values())

        if all_incumbent_feasible:
            cut_expressions = add_benders_cuts(incumbent_sp_constants, incumbent_sp_dv_arrays, incumbent_sp_feasibility, scenario_paths, master_nonant_vars_by_sp, theta_vars)
            for sp_id, cut_expression in cut_expressions.items():
                master_model.addConstr(cut_expression >= 0, name=f"incumbent_opt_cut_{sp_id}")

            master_model.update()

            log_file.write(f"Incumbent solution is feasible\n")

    if callback_flag:
        master_model.setParam('LazyConstraints', 1)
        master_model.setParam('PreCrush', 1)

        master_model._callback_data = {
            'iteration': 0,
            'lock': threading.Lock(),
            'log_file': log_file,
            'executors': executors,
            'scenario_paths': scenario_paths,
            'scenario_path_probabilities': scenario_path_probabilities,
            'best_upper_bound': float('inf'),
            'best_lower_bound': float('-inf'),
            'best_ub_solution': None,
            'nonant_vars': nonant_vars,
            'master_nonant_vars_by_sp': master_nonant_vars_by_sp,
            'master_dv_indices': master_dv_indices,
            'theta_vars': theta_vars,
            'scenario_path_keys': list(scenario_paths.keys()),
            'continuous_flag': continuous_flag,
            'valid_inequalities_flag': valid_inequalities_flag,
            'separation_data': separation_data,
            'tolerance': tolerance,
            'results_directory': results_directory,
            'total_iteration_time': total_iteration_time,
            'total_subproblem_time': total_subproblem_time,
            'total_valid_inequality_time': total_valid_inequality_time,
            'valid_inequalities_added': valid_inequalities_added,
            'feasibility_cut_iterations': feasibility_cut_iterations,
            'optimality_cut_iterations': optimality_cut_iterations
        }

        master_start_time = time.time()
        master_model.optimize(benders_callback)

        total_master_time = time.time() - master_start_time
        total_iteration_time = master_model._callback_data['total_iteration_time']
        total_subproblem_time = master_model._callback_data['total_subproblem_time']
        total_valid_inequality_time = master_model._callback_data['total_valid_inequality_time']
        valid_inequalities_added = master_model._callback_data['valid_inequalities_added']
        iteration = master_model._callback_data['iteration']
        feasibility_cut_iterations = master_model._callback_data['feasibility_cut_iterations']
        optimality_cut_iterations = master_model._callback_data['optimality_cut_iterations']
        best_upper_bound = master_model._callback_data['best_upper_bound']
        best_lower_bound = master_model._callback_data['best_lower_bound']
        best_ub_solution = master_model._callback_data['best_ub_solution']

    else:
        iteration = 0
        best_upper_bound = float('inf')
        best_lower_bound = float('-inf')
        best_ub_solution = None
        previous_nonant_solution = None

        prune_inactive_count = 10
        prune_tolerance = 1e-3
        scenario_path_ids = list(scenario_paths.keys())
        scenario_path_prob_array = np.array([scenario_path_probabilities[sp_id] for sp_id in scenario_path_ids], dtype=np.float64)
        all_master_vars = master_model.getVars()
        nontheta_variables = [not var.varName.startswith("theta") for var in all_master_vars]

        while True:
            iteration += 1
            master_start_time = time.time()
            master_model.optimize()
            master_execution_time = time.time() - master_start_time
            total_master_time += master_execution_time
            
            if continuous_flag:
                lower_bound = master_model.ObjVal
            else:
                lower_bound = master_model.ObjBound
            
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound

            if generated_cuts:
                generated_cuts_snapshot = list(generated_cuts.items())
                slacks = master_model.getAttr('Slack', [e['constraint'] for _, e in generated_cuts_snapshot])
                to_remove = []
                for (cut_name, entry), slack in zip(generated_cuts_snapshot, slacks):
                    if slack > prune_tolerance:
                        entry['inactive_count'] += 1
                        if entry['inactive_count'] >= prune_inactive_count:
                            master_model.remove(entry['constraint'])
                            to_remove.append(cut_name)
                    else:
                        entry['inactive_count'] = 0
                for cut_name in to_remove:
                    del generated_cuts[cut_name]

            nonant_solution_values = master_model.getAttr('X', nonant_vars)
            nonant_solution_array = np.array(nonant_solution_values, dtype=np.float64)

            subproblem_start_time = time.time()
            futures = {sp_id: executors[sp_id].submit(solve_subproblem, nonant_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_path_ids}
            subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
            subproblem_execution_time = time.time() - subproblem_start_time
            total_subproblem_time += subproblem_execution_time

            subproblem_objectives = {sp_id: r[0] for sp_id, r in subproblem_results.items()}
            subproblem_constants = {sp_id: r[1] for sp_id, r in subproblem_results.items()}
            subproblem_dv_arrays = {sp_id: r[2] for sp_id, r in subproblem_results.items()}
            subproblem_feasibility = {sp_id: r[3] for sp_id, r in subproblem_results.items()}
            subproblem_statuses = {sp_id: r[4] for sp_id, r in subproblem_results.items()}

            unexpected_statuses = [(sp_id, status) for sp_id, status in subproblem_statuses.items() if status != GRB.OPTIMAL and status != GRB.INFEASIBLE]
            if unexpected_statuses:
                with open(os.path.join(results_directory, 'SubproblemStatusLog.txt'), 'a') as status_file:
                    for sp_id, status in unexpected_statuses:
                        status_file.write(f"Iteration {iteration}: Subproblem {sp_id} status: {status}\n")

            all_feasible = all(subproblem_feasibility.values())

            if all_feasible:
                optimality_cut_iterations += 1
            else:
                feasibility_cut_iterations += 1

            sp_obj_array = np.array([subproblem_objectives[sp_id] for sp_id in scenario_path_ids], dtype=np.float64)

            theta_values = master_model.getAttr('X', theta_vars_values_list)
            theta_sum = np.dot(np.array(theta_values, dtype=np.float64), scenario_path_prob_array)
            upper_bound = master_model.ObjVal - theta_sum + np.dot(sp_obj_array, scenario_path_prob_array)
            cut_expressions = add_benders_cuts(subproblem_constants, subproblem_dv_arrays, subproblem_feasibility, scenario_paths, master_nonant_vars_by_sp, theta_vars)
            for sp_id, cut_expression in cut_expressions.items():
                cut_name = f'OptimalityCut{sp_id}_{iteration}' if all_feasible else f'FeasibilityCut{sp_id}_{iteration}'
                cut = master_model.addConstr(cut_expression >= 0, name=cut_name)
                generated_cuts[cut_name] = {'constraint': cut, 'inactive_count': 0}

            valid_inequality_derivation_time = 0
            valid_ineq_cut_expressions = None
            if not all_feasible and valid_inequalities_flag:
                valid_inequality_start_time = time.time()
                valid_ineq_cut_expressions = add_valid_inequalities(separation_data, subproblem_feasibility=subproblem_feasibility, master_model=master_model)
                for cut_name, cut_expression in valid_ineq_cut_expressions.items():
                    cut = master_model.addConstr(cut_expression >= 0, name=f'{cut_name}_{iteration}')
                    generated_cuts[f'{cut_name}_{iteration}'] = {'constraint': cut, 'inactive_count': 0}
                valid_inequality_derivation_time = time.time() - valid_inequality_start_time
                total_valid_inequality_time += valid_inequality_derivation_time
                valid_inequalities_added += len(valid_ineq_cut_expressions)

            if previous_nonant_solution is not None and not continuous_flag:
                if np.array_equal(nonant_solution_array, previous_nonant_solution):
                    current_mipgap = master_model.Params.MIPGap
                    new_mipgap = float(current_mipgap) * 0.5
                    master_model.setParam('MIPGap', new_mipgap)
            previous_nonant_solution = nonant_solution_array

            if all_feasible and upper_bound < best_upper_bound:
                best_upper_bound = upper_bound
                best_ub_var_values = master_model.getAttr('X', all_master_vars)
                best_ub_solution = {var.varName: val for var, val, keep in zip(all_master_vars, best_ub_var_values, nontheta_variables) if keep}

            gap = (best_upper_bound - best_lower_bound) / max(1e-6, best_upper_bound)
            log_lines = [
                '-' * 30,
                f"Iteration {iteration}:",
                f"Upper Bound: {best_upper_bound:.2f}",
                f"Lower Bound: {best_lower_bound:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds",
                f"Master Problem Execution Time: {master_execution_time:.2f} seconds"
            ]

            if valid_inequality_derivation_time != 0:
                log_lines.append(f"Valid Inequality Derivation Time: {valid_inequality_derivation_time:.2f} seconds")

            log_file.write('\n'.join(log_lines) + '\n')

            if gap < tolerance:
                break

    #lp_filename = os.path.join(results_directory, 'MasterModel.lp')
    #master_model.write(lp_filename)

    if not callback_flag:
        total_iteration_time = time.time() - total_iteration_time
    
    final_gap = (best_upper_bound - best_lower_bound) / max(1e-6, best_upper_bound)
    summary_lines = [
        '=' * 30,
        'Final Summary',
        f'Best Upper Bound: {best_upper_bound:.2f}',
        f'Final Lower Bound: {best_lower_bound:.2f}',
        f'Final Gap: {(100 * final_gap):.2f}%',
        f'Number of Iterations: {iteration}',
        f'Number of Iterations with Feasibility Cuts: {feasibility_cut_iterations}',
        f'Number of Iterations with Optimality Cuts: {optimality_cut_iterations}',
        f'Subproblem Time: {total_subproblem_time:.2f} seconds',
        f'Master Time: {total_master_time:.2f} seconds',
        f'Iteration Time: {total_iteration_time:.2f} seconds'
    ]

    if valid_inequalities_flag:
        summary_lines.append(f'Valid Inequality Time: {total_valid_inequality_time:.2f} seconds')
        summary_lines.append(f'Number of Valid Inequalities: {valid_inequalities_added}')

    log_file.write('\n'.join(summary_lines) + '\n')

    if incumbent_solve:
        best_solution_nonant_array = np.array([best_ub_solution[name] for name in nonant_var_names], dtype=np.float64)
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, best_solution_nonant_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}
        for future in futures.values():
            future.result()

        solution_dict = dict(best_ub_solution)
        for _, executor in executors.items():
            future = executor.submit(get_all_subproblem_solution)
            sub_vars = future.result()
            for var_name, var_value in sub_vars.items():
                if var_name not in solution_dict:
                    solution_dict[var_name] = var_value

        master_model.dispose()
        master_env.dispose()

        for executor in executors.values():
            executor.shutdown(wait=True)

        return solution_dict

    final_sol_file = os.path.join(results_directory, 'Results.sol')
    with open(final_sol_file, 'w') as f:
        for var_name, var_value in best_ub_solution.items():
            f.write(f"{var_name} {var_value}\n")

    best_solution_nonant_array = np.array([best_ub_solution[name] for name in nonant_var_names], dtype=np.float64)
    electricity_carry_values, heat_carry_values = write_final_subproblem_solutions(executors, best_solution_nonant_array, master_dv_indices, results_directory, scenario_paths, numStages, numSubperiods, numSubterms)

    for executor in executors.values():
        executor.shutdown(wait=True)

    if aggregated_subproblems_flag == False:
        operational_model, operational_env = OperationalNonanticipativityModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor, aggregated_subproblems_flag)

        vars_to_fix = []
        bounds = []
        
        for var_name, var_value in best_ub_solution.items():
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
        log_file.write(f'Nonanticipativity Model Solve Time: {operational_model.Runtime:.2f} seconds\n')

        with open(final_sol_file, 'a') as f:
            for var in operational_model.getVars():
                if not var.varName.startswith('plus_'):
                    f.write(f"{var.varName} {var.X}\n")

        operational_model.dispose()
        operational_env.dispose()

    master_model.dispose()
    master_env.dispose()