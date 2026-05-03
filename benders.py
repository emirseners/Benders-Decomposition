from scipy.sparse import csr_matrix, vstack as sparse_vstack
from gurobipy import GRB, LinExpr
from collections import deque
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

def _compute_master_scaling_statistics(master_model):
    A = master_model.getA()
    abs_coefs = np.abs(A.data)

    geo_mean = np.exp(np.mean(np.log(abs_coefs)))
    max_coef = np.max(abs_coefs)
    min_coef = np.min(abs_coefs)
    coef_range_ratio = max_coef / min_coef

    return {
        'geo_mean': geo_mean,
        'max_coef': max_coef,
        'min_coef': min_coef,
        'coef_range_ratio': coef_range_ratio,
    }

def get_master_var_names():
    return _cached_worker_model._master_var_names

def solve_subproblem(master_solution):
    _worker_model = _cached_worker_model
    master_solution = np.where(np.abs(master_solution) < 1e-12, 0.0, master_solution)

    delta = np.asarray(_worker_model._A_coupling @ master_solution)
    new_rhs = _worker_model._coupling_rhs - delta
    _worker_model.setAttr('RHS', _worker_model._coupling_constrs, new_rhs.tolist())

    _worker_model.optimize()

    status = _worker_model.status
    is_optimal = status == GRB.OPTIMAL

    if not is_optimal:
        assert status == GRB.INFEASIBLE, f"Unexpected subproblem status: {status}"

    if is_optimal:
        objective_value = _worker_model.objVal
        dual_values = np.array(_worker_model.getAttr('Pi', _worker_model._all_constrs))
        dual_coefs = np.asarray(dual_values @ _worker_model._A_master)
        cut_intercept = objective_value + np.dot(dual_coefs, master_solution)
    else:
        objective_value = float('inf')
        farkas_values = np.array(_worker_model.getAttr('FarkasDual', _worker_model._all_constrs))
        cut_intercept = -np.dot(farkas_values, _worker_model._all_rhs)
        dual_coefs = np.asarray(farkas_values @ _worker_model._A_master)
        np.negative(dual_coefs, out=dual_coefs)

    return objective_value, cut_intercept, dual_coefs, is_optimal

def _compute_scale_factor(cut_values, master_scaling_stats):
    abs_values = np.abs(cut_values)
    abs_values = abs_values[abs_values > 0]
    if len(abs_values) == 0:
        return 1.0

    cut_geo_mean = np.exp(np.mean(np.log(abs_values)))
    cut_max = abs_values.max()
    cut_min = abs_values.min()
    cut_range = cut_max / cut_min

    scale_factor = master_scaling_stats['geo_mean'] / cut_geo_mean

    master_max = master_scaling_stats['max_coef']
    master_min = master_scaling_stats['min_coef']

    if cut_range <= master_scaling_stats['coef_range_ratio']:
        if cut_max * scale_factor > master_max:
            scale_factor = master_max / cut_max
        elif cut_min * scale_factor < master_min:
            scale_factor = master_min / cut_min
    else:
        if cut_max * scale_factor > master_max:
            scale_factor = master_max / cut_max

    return scale_factor

def add_cut_lp_model(lp_model, lp_vars, cut_var_indices, cut_coefs, cut_constant, name):
    lp_expr = LinExpr(cut_coefs.tolist(), lp_vars[cut_var_indices].tolist())
    lp_expr.addConstant(cut_constant)
    return lp_model.addConstr(lp_expr >= 0, name=name)

def register_cut(name, expr, scale_factor, cut_var_indices, cut_coefs, cut_constant, master_model, lp_master_model, lp_all_vars, generated_cuts_data):
    constr = master_model.addConstr(expr >= 0, name=name)
    lp_constr = add_cut_lp_model(lp_master_model, lp_all_vars, cut_var_indices, cut_coefs, cut_constant, name) if lp_master_model is not None else None
    generated_cuts_data['names'].append(name)
    generated_cuts_data['constraints'].append(constr)
    generated_cuts_data['lp_constraints'].append(lp_constr)
    generated_cuts_data['scale_factors'].append(scale_factor)
    generated_cuts_data['var_indices'].append(cut_var_indices)
    generated_cuts_data['coefs'].append(cut_coefs)
    generated_cuts_data['constants'].append(cut_constant)

def build_cut_matrix_rows(var_indices_list, coefs_list, constants_list, num_vars):
    n = len(var_indices_list)
    rows, cols, data = [], [], []
    for i in range(n):
        indices, coefs = var_indices_list[i], coefs_list[i]
        rows.append(np.full(len(indices), i, dtype=np.int32))
        cols.append(indices)
        data.append(coefs)
    matrix = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, num_vars))
    return matrix, np.array(constants_list, dtype=np.float64)

def build_cut_linexpr(significant_dual_coefs, significant_idx, scale_factor, master_vars, master_var_model_indices, cut_intercept, theta_var=None, theta_model_index=None):
    expr = LinExpr()
    cut_constant = -cut_intercept * scale_factor
    n = len(significant_idx)
    if n > 0:
        scaled_coefs = significant_dual_coefs * scale_factor
        coefficients = scaled_coefs.tolist()
        vars_list = master_vars[significant_idx].tolist()
        if theta_var is not None:
            coefficients.append(scale_factor)
            vars_list.append(theta_var)
            cut_var_indices = np.empty(n + 1, dtype=np.int32)
            cut_var_indices[:n] = master_var_model_indices[significant_idx]
            cut_var_indices[n] = theta_model_index
            cut_coefs = np.empty(n + 1, dtype=np.float64)
            cut_coefs[:n] = scaled_coefs
            cut_coefs[n] = scale_factor
        else:
            cut_var_indices = master_var_model_indices[significant_idx]
            cut_coefs = scaled_coefs
        expr.addTerms(coefficients, vars_list)
    elif theta_var is not None:
        expr.addTerms([scale_factor], [theta_var])
        cut_var_indices = np.array([theta_model_index], dtype=np.int32)
        cut_coefs = np.array([scale_factor], dtype=np.float64)
    else:
        cut_var_indices = np.array([], dtype=np.int32)
        cut_coefs = np.array([], dtype=np.float64)
    expr.addConstant(cut_constant)
    return expr, cut_var_indices, cut_coefs, cut_constant

def add_benders_cuts(subproblem_cut_intercepts, subproblem_dual_coefs, subproblem_feasibility, master_vars_by_sp, theta_vars, master_scaling_stats, master_model_indices, theta_model_indices, all_var_values):
    cut_expressions = {}

    for sp_id, is_feasible in subproblem_feasibility.items():
        cut_intercept = subproblem_cut_intercepts[sp_id]
        dual_coefs = subproblem_dual_coefs[sp_id]
        #significant_idx = np.nonzero(dual_coefs)[0]
        abs_dual_coefs = np.abs(dual_coefs)
        max_abs = abs_dual_coefs.max() if abs_dual_coefs.size else 0.0
        significant_idx = np.where(abs_dual_coefs > max(1e-12, 1e-10 * max_abs))[0]
        significant_dual_coefs = dual_coefs[significant_idx]
        length_significant = len(significant_idx)
        if is_feasible:
            cut_values = np.empty(length_significant + 1, dtype=np.float64)
            cut_values[0] = 1.0
            cut_values[1:] = significant_dual_coefs
            scale_factor = _compute_scale_factor(cut_values, master_scaling_stats)
            expr, cut_var_indices, cut_coefs, cut_constant = build_cut_linexpr(significant_dual_coefs, significant_idx, scale_factor, master_vars_by_sp[sp_id], master_model_indices[sp_id], cut_intercept, theta_vars[sp_id], theta_model_indices[sp_id])
        else:
            scale_factor = _compute_scale_factor(significant_dual_coefs, master_scaling_stats)
            expr, cut_var_indices, cut_coefs, cut_constant = build_cut_linexpr(significant_dual_coefs, significant_idx, scale_factor, master_vars_by_sp[sp_id], master_model_indices[sp_id], cut_intercept)

        slack = np.dot(cut_coefs, all_var_values[cut_var_indices]) + cut_constant
        if slack >= -1e-6 * scale_factor:
            continue

        cut_expressions[sp_id] = (expr, scale_factor, cut_var_indices, cut_coefs, cut_constant)

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

def _build_electricity_cut(sp_id, scenario_path_separation_data, electricity_storage_const, q_lb_e, q_ub_e, min_sum_e, master_scaling_stats):
    if min_sum_e + electricity_storage_const >= 0:
        return None, None, None, None, None, None

    electricity_demand_sum = scenario_path_separation_data['electricity_demand_cumsum'][q_ub_e] - scenario_path_separation_data['electricity_demand_cumsum'][q_lb_e - 1]
    summed_coefs = scenario_path_separation_data['elec_coef_cumsum'][q_ub_e] - scenario_path_separation_data['elec_coef_cumsum'][q_lb_e - 1]

    combined_coefs = np.concatenate([summed_coefs, scenario_path_separation_data['elec_storage_coefs']])
    cut_values = combined_coefs[np.nonzero(combined_coefs)[0]]
    scale_factor = _compute_scale_factor(cut_values, master_scaling_stats)

    scaled_summed = summed_coefs * scale_factor
    scaled_storage = scenario_path_separation_data['elec_storage_coefs'] * scale_factor

    cut_name = f'ValidInequality_Electricity_SP{sp_id}_q{q_lb_e}_{q_ub_e}'
    expr = LinExpr()
    expr.addTerms(scaled_summed.tolist(), scenario_path_separation_data['elec_gen_vars'])
    expr.addTerms(scaled_storage.tolist(), scenario_path_separation_data['elec_storage_vars'])
    cut_constant = -electricity_demand_sum * scale_factor
    expr.addConstant(cut_constant)

    cut_var_indices = scenario_path_separation_data['elec_combined_var_indices']
    cut_coefs = np.concatenate([scaled_summed, scaled_storage])

    return cut_name, expr, scale_factor, cut_var_indices, cut_coefs, cut_constant

def _build_heat_cut(sp_id, scenario_path_separation_data, heat_storage_const, q_lb_h, q_ub_h, min_sum_h, master_scaling_stats):
    if min_sum_h + heat_storage_const >= 0:
        return None, None, None, None, None, None

    heat_demand_sum = scenario_path_separation_data['heat_demand_cumsum'][q_ub_h] - scenario_path_separation_data['heat_demand_cumsum'][q_lb_h - 1]
    num_subterms = q_ub_h - q_lb_h + 1
    summed_coefs = scenario_path_separation_data['heat_coef_cumsum'][q_ub_h] - scenario_path_separation_data['heat_coef_cumsum'][q_lb_h - 1]

    combined_coefs = np.concatenate([summed_coefs, scenario_path_separation_data['heat_transfer_coefs'] * num_subterms, scenario_path_separation_data['heat_storage_coefs']])
    cut_values = combined_coefs[np.nonzero(combined_coefs)[0]]
    scale_factor = _compute_scale_factor(cut_values, master_scaling_stats)

    scaled_summed = summed_coefs * scale_factor
    scaled_transfer = scenario_path_separation_data['heat_transfer_coefs'] * (scale_factor * num_subterms)
    scaled_storage = scenario_path_separation_data['heat_storage_coefs'] * scale_factor

    cut_name = f'ValidIneq_Heat_SP{sp_id}_q{q_lb_h}_{q_ub_h}'
    expr = LinExpr()
    expr.addTerms(scaled_summed.tolist(), scenario_path_separation_data['heat_gen_vars'])
    expr.addTerms(scaled_transfer.tolist(), scenario_path_separation_data['heat_transfer_vars'])
    expr.addTerms(scaled_storage.tolist(), scenario_path_separation_data['heat_storage_vars'])
    cut_constant = -heat_demand_sum * scale_factor
    expr.addConstant(cut_constant)

    cut_var_indices = scenario_path_separation_data['heat_combined_var_indices']
    cut_coefs = np.concatenate([scaled_summed, scaled_transfer, scaled_storage])

    return cut_name, expr, scale_factor, cut_var_indices, cut_coefs, cut_constant

def add_valid_inequalities(separation_data, subproblem_feasibility=None, callback_flag=False, master_model=None, all_var_values=None, initial_iteration=False, numSubterms=None, scenario_paths=None, master_scaling_stats=None):
    cut_expressions = {}

    if initial_iteration:
        for sp_id in scenario_paths.keys():
            scenario_path_separation_data = separation_data[sp_id]

            electricity_demand_sum = scenario_path_separation_data['electricity_demand_cumsum'][numSubterms]
            elec_summed_coefs = scenario_path_separation_data['elec_coef_cumsum'][numSubterms]
            elec_combined_coefs = np.concatenate([elec_summed_coefs, scenario_path_separation_data['elec_storage_coefs']])
            elec_cut_values = elec_combined_coefs[np.nonzero(elec_combined_coefs)[0]]
            scale_factor = _compute_scale_factor(elec_cut_values, master_scaling_stats)
            scaled_elec_summed = elec_summed_coefs * scale_factor
            scaled_elec_storage = scenario_path_separation_data['elec_storage_coefs'] * scale_factor
            expr_e = LinExpr()
            expr_e.addTerms(scaled_elec_summed.tolist(), scenario_path_separation_data['elec_gen_vars'])
            expr_e.addTerms(scaled_elec_storage.tolist(), scenario_path_separation_data['elec_storage_vars'])
            elec_cut_constant = -electricity_demand_sum * scale_factor
            expr_e.addConstant(elec_cut_constant)
            elec_var_indices = scenario_path_separation_data['elec_combined_var_indices']
            elec_coefs = np.concatenate([scaled_elec_summed, scaled_elec_storage])
            cut_expressions[f'ValidInequality_Electricity_SP{sp_id}_q{1}_{numSubterms}'] = (expr_e, scale_factor, elec_var_indices, elec_coefs, elec_cut_constant)

            heat_demand_sum = scenario_path_separation_data['heat_demand_cumsum'][numSubterms]
            heat_summed_coefs = scenario_path_separation_data['heat_coef_cumsum'][numSubterms]
            heat_combined_coefs = np.concatenate([heat_summed_coefs, scenario_path_separation_data['heat_transfer_coefs'] * numSubterms, scenario_path_separation_data['heat_storage_coefs']])
            heat_cut_values = heat_combined_coefs[np.nonzero(heat_combined_coefs)[0]]
            scale_factor = _compute_scale_factor(heat_cut_values, master_scaling_stats)
            scaled_heat_summed = heat_summed_coefs * scale_factor
            scaled_heat_transfer = scenario_path_separation_data['heat_transfer_coefs'] * (scale_factor * numSubterms)
            scaled_heat_storage = scenario_path_separation_data['heat_storage_coefs'] * scale_factor
            expr_h = LinExpr()
            expr_h.addTerms(scaled_heat_summed.tolist(), scenario_path_separation_data['heat_gen_vars'])
            expr_h.addTerms(scaled_heat_transfer.tolist(), scenario_path_separation_data['heat_transfer_vars'])
            expr_h.addTerms(scaled_heat_storage.tolist(), scenario_path_separation_data['heat_storage_vars'])
            heat_cut_constant = -heat_demand_sum * scale_factor
            expr_h.addConstant(heat_cut_constant)
            heat_var_indices = scenario_path_separation_data['heat_combined_var_indices']
            heat_coefs = np.concatenate([scaled_heat_summed, scaled_heat_transfer, scaled_heat_storage])
            cut_expressions[f'ValidIneq_Heat_SP{sp_id}_q{1}_{numSubterms}'] = (expr_h, scale_factor, heat_var_indices, heat_coefs, heat_cut_constant)

        return cut_expressions

    for sp_id, is_feasible in subproblem_feasibility.items():
        if is_feasible:
            continue

        scenario_path_separation_data = separation_data[sp_id]
        electricity_demand = scenario_path_separation_data['electricity_demand']
        heat_demand = scenario_path_separation_data['heat_demand']
        elec_gen_coef_matrix = scenario_path_separation_data['elec_gen_coef_matrix']
        heat_gen_coef_matrix = scenario_path_separation_data['heat_gen_coef_matrix']

        if callback_flag:
            elec_gen_vals = np.array(master_model.cbGetSolution(scenario_path_separation_data['elec_gen_vars']))
            heat_gen_vals = np.array(master_model.cbGetSolution(scenario_path_separation_data['heat_gen_vars']))
            elec_storage_vals = np.array(master_model.cbGetSolution(scenario_path_separation_data['elec_storage_vars']))
            heat_transfer_vals = np.array(master_model.cbGetSolution(scenario_path_separation_data['heat_transfer_vars']))
            heat_storage_vals = np.array(master_model.cbGetSolution(scenario_path_separation_data['heat_storage_vars']))
        else:
            elec_gen_vals = all_var_values[scenario_path_separation_data['elec_gen_var_indices']]
            heat_gen_vals = all_var_values[scenario_path_separation_data['heat_gen_var_indices']]
            elec_storage_vals = all_var_values[scenario_path_separation_data['elec_storage_var_indices']]
            heat_transfer_vals = all_var_values[scenario_path_separation_data['heat_transfer_var_indices']]
            heat_storage_vals = all_var_values[scenario_path_separation_data['heat_storage_var_indices']]

        electricity_storage_const = np.dot(scenario_path_separation_data['elec_storage_coefs'], elec_storage_vals)
        heat_transfer_per_subperiod = np.dot(scenario_path_separation_data['heat_transfer_coefs'], heat_transfer_vals)
        heat_storage_const = np.dot(scenario_path_separation_data['heat_storage_coefs'], heat_storage_vals)

        electricity_contiguous_array = elec_gen_coef_matrix @ elec_gen_vals - electricity_demand
        heat_contiguous_array = (heat_gen_coef_matrix @ heat_gen_vals) + heat_transfer_per_subperiod - heat_demand

        min_sum_e, q_lb_e, q_ub_e = minimum_sum_contiguous_subarray(np.ascontiguousarray(electricity_contiguous_array))
        min_sum_h, q_lb_h, q_ub_h = minimum_sum_contiguous_subarray(np.ascontiguousarray(heat_contiguous_array))

        elec_cut_name, elec_cut_expr, elec_scale_factor, elec_var_indices, elec_coefs, elec_cut_constant = _build_electricity_cut(sp_id, scenario_path_separation_data, electricity_storage_const, q_lb_e, q_ub_e, min_sum_e, master_scaling_stats)
        heat_cut_name, heat_cut_expr, heat_scale_factor, heat_var_indices, heat_coefs, heat_cut_constant = _build_heat_cut(sp_id, scenario_path_separation_data, heat_storage_const, q_lb_h, q_ub_h, min_sum_h, master_scaling_stats)

        if elec_cut_name is not None:
            cut_expressions[elec_cut_name] = (elec_cut_expr, elec_scale_factor, elec_var_indices, elec_coefs, elec_cut_constant)
        if heat_cut_name is not None:
            cut_expressions[heat_cut_name] = (heat_cut_expr, heat_scale_factor, heat_var_indices, heat_coefs, heat_cut_constant)

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

    leaf_index = f'[{(numStages-1) * numSubperiods},{numSubterms}]'

    e_carry_var = _worker_model.getVarByName(f'electricitycarry_{leaf_parent_node_id}{leaf_index}')
    h_carry_var = _worker_model.getVarByName(f'heatcarry_{leaf_parent_node_id}{leaf_index}')
    
    return leaf_vars, (e_carry_var.varName, e_carry_var.X), (h_carry_var.varName, h_carry_var.X)

def get_all_subproblem_solution():
    _worker_model = _cached_worker_model
    all_vars = {}
    for var in _worker_model.getVars():
        if not var.varName.startswith('plus_'):
            all_vars[var.varName] = var.X
    return all_vars

def write_final_subproblem_solutions(executors, master_solution_array, master_dv_indices, results_directory, scenario_paths, numStages, numSubperiods, numSubterms):
    futures = {sp_id: executors[sp_id].submit(solve_subproblem, master_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}
    
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

        master_values = model.cbGetSolution(call_back_data['master_vars'])
        master_solution_array = np.array(master_values, dtype=np.float64)
        master_dv_indices = call_back_data['master_dv_indices']

        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if call_back_data['continuous_flag']:
            lower_bound = current_obj
        else:
            lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        subproblem_start_time = time.time()
        futures = {sp_id: call_back_data['executors'][sp_id].submit(solve_subproblem, master_solution_array[master_dv_indices[sp_id]]) for sp_id in call_back_data['scenario_paths'].keys()}
        subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
        subproblem_execution_time = time.time() - subproblem_start_time

        subproblem_objectives = {}
        subproblem_cut_intercepts = {}
        subproblem_dual_coefs = {}
        subproblem_feasibility = {}
        for sp_id, (obj, cut_intercept, dual_coef, is_feasible) in subproblem_results.items():
            subproblem_objectives[sp_id] = obj
            subproblem_cut_intercepts[sp_id] = cut_intercept
            subproblem_dual_coefs[sp_id] = dual_coef
            subproblem_feasibility[sp_id] = is_feasible

        if 'theta_vars_list' not in call_back_data:
            call_back_data['theta_vars_list'] = list(call_back_data['theta_vars'].values())
        theta_values = model.cbGetSolution(call_back_data['theta_vars_list'])
        scenario_path_probabilities = call_back_data['scenario_path_probabilities']
        scenario_path_ids = call_back_data['scenario_path_ids']
        theta_sum = sum(theta_value * scenario_path_probabilities[sp_id] for theta_value, sp_id in zip(theta_values, scenario_path_ids))
        subproblem_obj_sum = sum(subproblem_objectives[sp_id] * scenario_path_probabilities[sp_id] for sp_id in scenario_path_ids)
        upper_bound = current_obj - theta_sum + subproblem_obj_sum
        all_var_values = np.array(model.cbGetSolution(call_back_data['all_vars']), dtype=np.float64)
        cut_expressions = add_benders_cuts(subproblem_cut_intercepts, subproblem_dual_coefs, subproblem_feasibility, call_back_data['master_vars_by_sp'], call_back_data['theta_vars'], call_back_data['master_scaling_stats'], call_back_data['master_model_indices'], call_back_data['theta_model_indices'], all_var_values)

        for cut_expression, *_ in cut_expressions.values():
            model.cbLazy(cut_expression >= 0)

        all_feasible = all(subproblem_feasibility.values())

        valid_inequality_derivation_time = 0
        num_valid_inequalities = 0
        if not all_feasible and call_back_data['valid_inequalities_flag']:
            valid_inequality_start_time = time.time()
            valid_ineq_cut_expressions = add_valid_inequalities(call_back_data['separation_data'], subproblem_feasibility=subproblem_feasibility, callback_flag=True, master_model=model, master_scaling_stats=call_back_data['master_scaling_stats'])
            num_valid_inequalities = len(valid_ineq_cut_expressions)
            for _, (cut_expression, *_) in valid_ineq_cut_expressions.items():
                model.cbLazy(cut_expression >= 0)
            valid_inequality_derivation_time = time.time() - valid_inequality_start_time

        with call_back_data['lock']:
            call_back_data['total_subproblem_time'] += subproblem_execution_time

            optimality_cut_amount = sum(1 for sp_id in cut_expressions if subproblem_feasibility[sp_id])
            feasibility_cut_amount = len(cut_expressions) - optimality_cut_amount
            call_back_data['optimality_cuts_added'] += optimality_cut_amount
            call_back_data['feasibility_cuts_added'] += feasibility_cut_amount

            if valid_inequality_derivation_time > 0:
                call_back_data['total_valid_inequality_time'] += valid_inequality_derivation_time
                call_back_data['valid_inequalities_added'] += num_valid_inequalities

            if all_feasible and upper_bound < call_back_data['best_upper_bound']:
                call_back_data['best_upper_bound'] = upper_bound
                call_back_data['best_ub_solution'] = {var.varName: val for var, val in zip(call_back_data['all_vars'], all_var_values) if not var.varName.startswith("theta")}

            if lower_bound > call_back_data['best_lower_bound']:
                call_back_data['best_lower_bound'] = lower_bound

            gap = (call_back_data['best_upper_bound'] - call_back_data['best_lower_bound']) / max(1e-6, call_back_data['best_upper_bound'])

            log_lines = [
                '-' * 30,
                f"Iteration {call_back_data['iteration']}:",
                f"Upper Bound: {call_back_data['best_upper_bound']:.2f}",
                f"Lower Bound: {call_back_data['best_lower_bound']:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Number of Optimality Cuts: {optimality_cut_amount}",
                f"Number of Feasibility Cuts: {feasibility_cut_amount}",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds",
                f"Iteration Time: {time.time() - iteration_start_time:.2f} seconds"
            ]

            call_back_data['log_file'].write('\n'.join(log_lines) + '\n')
            call_back_data['total_iteration_time'] += time.time() - iteration_start_time

            if gap < call_back_data['tolerance']:
                model.terminate()

def setup_master_variable_mappings(master_model, scenario_paths, sp_master_var_names, separation_data):
    all_gurobi_vars = master_model.getVars()
    var_cache = {var.varName: var for var in all_gurobi_vars}

    master_vars = []
    master_var_names = []
    nontheta_var_indices = []
    for i, var in enumerate(all_gurobi_vars):
        if not var.varName.startswith("theta"):
            master_vars.append(var)
            master_var_names.append(var.varName)
            nontheta_var_indices.append(i)
    nontheta_var_indices = np.array(nontheta_var_indices, dtype=np.int32)

    name_to_idx = {name: i for i, name in enumerate(master_var_names)}

    master_vars_by_sp = {}
    master_dv_indices = {}
    master_model_indices = {}
    for sp_id in scenario_paths:
        names = sp_master_var_names[sp_id]
        vars_list = [var_cache[name] for name in names]
        master_vars_by_sp[sp_id] = np.array(vars_list, dtype=object)
        master_dv_indices[sp_id] = np.array([name_to_idx[name] for name in names])
        master_model_indices[sp_id] = np.array([v.index for v in vars_list], dtype=np.int32)

    theta_vars = {sp_id: var_cache[f"theta[{sp_id}]"] for sp_id in scenario_paths}
    theta_var_list = list(theta_vars.values())
    theta_var_indices = np.array([v.index for v in theta_var_list], dtype=np.int32)
    theta_model_indices = {sp_id: theta_vars[sp_id].index for sp_id in scenario_paths}

    if separation_data is not None:
        for _, scenario_path_data in separation_data.items():
            scenario_path_data['elec_gen_vars'] = [var_cache[name] for name in scenario_path_data['elec_gen_var_names']]
            scenario_path_data['heat_gen_vars'] = [var_cache[name] for name in scenario_path_data['heat_gen_var_names']]
            scenario_path_data['elec_storage_vars'] = [var_cache[name] for name in scenario_path_data['elec_storage_var_names']]
            scenario_path_data['heat_storage_vars'] = [var_cache[name] for name in scenario_path_data['heat_storage_var_names']]
            scenario_path_data['heat_transfer_vars'] = [var_cache[name] for name in scenario_path_data['heat_transfer_var_names']]
            scenario_path_data['elec_storage_coefs'] = np.array(scenario_path_data['elec_storage_coefs'], dtype=np.float64)
            scenario_path_data['heat_storage_coefs'] = np.array(scenario_path_data['heat_storage_coefs'], dtype=np.float64)
            scenario_path_data['heat_transfer_coefs'] = np.array(scenario_path_data['heat_transfer_coefs'], dtype=np.float64)
            scenario_path_data['elec_gen_var_indices'] = np.array([v.index for v in scenario_path_data['elec_gen_vars']], dtype=np.int32)
            scenario_path_data['heat_gen_var_indices'] = np.array([v.index for v in scenario_path_data['heat_gen_vars']], dtype=np.int32)
            scenario_path_data['elec_storage_var_indices'] = np.array([v.index for v in scenario_path_data['elec_storage_vars']], dtype=np.int32)
            scenario_path_data['heat_storage_var_indices'] = np.array([v.index for v in scenario_path_data['heat_storage_vars']], dtype=np.int32)
            scenario_path_data['heat_transfer_var_indices'] = np.array([v.index for v in scenario_path_data['heat_transfer_vars']], dtype=np.int32)
            scenario_path_data['elec_combined_var_indices'] = np.concatenate([scenario_path_data['elec_gen_var_indices'], scenario_path_data['elec_storage_var_indices']])
            scenario_path_data['heat_combined_var_indices'] = np.concatenate([scenario_path_data['heat_gen_var_indices'], scenario_path_data['heat_transfer_var_indices'], scenario_path_data['heat_storage_var_indices']])

    return master_vars, master_var_names, master_vars_by_sp, master_dv_indices, theta_vars, master_model_indices, theta_model_indices, nontheta_var_indices, theta_var_indices

def run_benders(numStages, numSubperiods, numSubterms, scenarioTree, initial_tech, emission_limits, electricity_demand, heat_demand,
                budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, log_file, discount_factor, scenario_paths, 
                scenario_path_probabilities, tolerance, benders_without_feasibility_flag, aggregated_subproblems_flag, callback_flag, 
                continuous_flag, valid_inequalities_flag, master_threads, threads_per_worker, incumbent_solution, incumbent_solve=False):
    
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

    sp_master_var_names_futures = {sp_id: executors[sp_id].submit(get_master_var_names) for sp_id in scenario_paths}

    master_model, master_env, separation_data = MasterProblemModel(scenarioTree, emission_limits, electricity_demand, heat_demand, initial_tech, budget, electricity_purchasing_cost, heat_purchasing_cost, results_directory, master_threads, discount_factor, scenario_paths, scenario_path_probabilities, continuous_flag, valid_inequalities_flag, tolerance)

    master_scaling_stats = _compute_master_scaling_statistics(master_model)

    lp_all_vars = None
    lp_master_model = None

    sp_master_var_names = {sp_id: future.result() for sp_id, future in sp_master_var_names_futures.items()}
    master_vars, master_var_names, master_vars_by_sp, master_dv_indices, theta_vars, master_model_indices, theta_model_indices, nontheta_var_indices, theta_var_indices = setup_master_variable_mappings(master_model, scenario_paths, sp_master_var_names, separation_data)

    generated_cuts_data = {'names': [], 'constraints': [], 'lp_constraints': [], 'scale_factors': [], 'var_indices': [], 'coefs': [], 'constants': []}
    cut_inactive_counts = np.empty(0, dtype=np.int32)

    pruned_cut_data = {'names': [], 'var_indices': [], 'coefs': [], 'constants': [], 'scale_factors': []}
    pruned_cuts_matrix = None
    pruned_cuts_constants_array = None
    total_readded = 0

    if valid_inequalities_flag:
        valid_ineq_cut_expressions = add_valid_inequalities(separation_data, initial_iteration=True, numSubterms=numSubterms, scenario_paths=scenario_paths, master_scaling_stats=master_scaling_stats)
        for cut_name, (cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant) in valid_ineq_cut_expressions.items():
            register_cut(f'{cut_name}_{0}', cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant, master_model, lp_master_model, lp_all_vars, generated_cuts_data)
        cut_inactive_counts = np.zeros(len(generated_cuts_data['names']), dtype=np.int32)

    total_master_time = 0
    total_iteration_time = 0 if callback_flag else time.time()
    total_subproblem_time = 0
    total_valid_inequality_time = 0

    feasibility_cuts_added = 0
    optimality_cuts_added = 0
    valid_inequalities_added = len(valid_ineq_cut_expressions) if valid_inequalities_flag else 0

    if incumbent_solution is not None:
        incumbent_master_array = np.array([incumbent_solution[name] for name in master_var_names], dtype=np.float64)
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, incumbent_master_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}

        incumbent_sp_results = {sp_id: future.result() for sp_id, future in futures.items()}
        incumbent_sp_cut_intercepts = {}
        incumbent_sp_dual_coefs = {}
        incumbent_sp_feasibility = {}
        for sp_id, (_, cut_intercept, dual_coef, is_feasible) in incumbent_sp_results.items():
            incumbent_sp_cut_intercepts[sp_id] = cut_intercept
            incumbent_sp_dual_coefs[sp_id] = dual_coef
            incumbent_sp_feasibility[sp_id] = is_feasible

        all_incumbent_feasible = all(incumbent_sp_feasibility.values())

        if all_incumbent_feasible:
            incumbent_all_var_values = np.zeros(len(nontheta_var_indices) + len(theta_var_indices), dtype=np.float64)
            incumbent_all_var_values[nontheta_var_indices] = incumbent_master_array
            cut_expressions = add_benders_cuts(incumbent_sp_cut_intercepts, incumbent_sp_dual_coefs, incumbent_sp_feasibility, master_vars_by_sp, theta_vars, master_scaling_stats, master_model_indices, theta_model_indices, incumbent_all_var_values)
            for sp_id, (cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant) in cut_expressions.items():
                master_model.addConstr(cut_expression >= 0, name=f"incumbent_opt_cut_{sp_id}")
                if lp_master_model is not None:
                    add_cut_lp_model(lp_master_model, lp_all_vars, cut_var_indices, cut_coefs, cut_constant, f"incumbent_opt_cut_{sp_id}")

            log_file.write(f"Incumbent solution is feasible\n")

    if callback_flag:
        master_model.setParam('LazyConstraints', 1)
        master_model.setParam('PreCrush', 1)

        master_model._callback_data = {'iteration': 0, 'lock': threading.Lock(), 'log_file': log_file, 'executors': executors, 'scenario_paths': scenario_paths,
            'scenario_path_probabilities': scenario_path_probabilities, 'best_upper_bound': float('inf'), 'best_lower_bound': float('-inf'), 'best_ub_solution': None,
            'master_vars': master_vars, 'master_vars_by_sp': master_vars_by_sp, 'master_dv_indices': master_dv_indices, 'theta_vars': theta_vars, 
            'scenario_path_ids': list(scenario_paths.keys()), 'continuous_flag': continuous_flag, 'valid_inequalities_flag': valid_inequalities_flag,
            'separation_data': separation_data, 'tolerance': tolerance, 'total_iteration_time': total_iteration_time, 'total_subproblem_time': total_subproblem_time,
            'total_valid_inequality_time': total_valid_inequality_time, 'valid_inequalities_added': valid_inequalities_added, 'feasibility_cuts_added': feasibility_cuts_added,
            'optimality_cuts_added': optimality_cuts_added, 'master_scaling_stats': master_scaling_stats, 'master_model_indices': master_model_indices,
            'theta_model_indices': theta_model_indices, 'all_vars': master_model.getVars()
        }

        master_start_time = time.time()
        master_model.optimize(benders_callback)

        total_master_time = time.time() - master_start_time
        total_iteration_time = master_model._callback_data['total_iteration_time']
        total_subproblem_time = master_model._callback_data['total_subproblem_time']
        total_valid_inequality_time = master_model._callback_data['total_valid_inequality_time']
        valid_inequalities_added = master_model._callback_data['valid_inequalities_added']
        iteration = master_model._callback_data['iteration']
        feasibility_cuts_added = master_model._callback_data['feasibility_cuts_added']
        optimality_cuts_added = master_model._callback_data['optimality_cuts_added']
        best_upper_bound = master_model._callback_data['best_upper_bound']
        best_lower_bound = master_model._callback_data['best_lower_bound']
        best_ub_solution = master_model._callback_data['best_ub_solution']

    else:
        iteration = 0
        best_upper_bound = float('inf')
        best_lower_bound = float('-inf')
        best_ub_solution = None
        previous_master_solution = None

        prune_inactive_count = 100
        prune_percentile = 75
        scenario_path_ids = list(scenario_paths.keys())
        scenario_path_prob_array = np.array([scenario_path_probabilities[sp_id] for sp_id in scenario_path_ids], dtype=np.float64)
        all_master_vars = master_model.getVars()
        all_master_vars_array = np.array(all_master_vars, dtype=object)
        num_master_vars = len(all_master_vars)
        nontheta_mask = [not var.varName.startswith("theta") for var in all_master_vars]
        sp_objectives_array = np.empty(len(scenario_path_ids), dtype=np.float64)

        lp_phase = False
        original_variable_types = {}
        if not continuous_flag:
            integer_vars = [var for var in master_model.getVars() if var.vType != GRB.CONTINUOUS]
            original_variable_types = {var: var.vType for var in integer_vars}
            for var in integer_vars:
                var.setAttr('VType', GRB.CONTINUOUS)
            master_model.update()
            lp_phase = True

        best_lb_array = deque([None]*51, maxlen=51)
        all_ub_found = []

        while True:
            iteration += 1
            master_model.optimize()
            master_execution_time = master_model.Runtime
            total_master_time += master_execution_time

            all_var_values = np.array(master_model.getAttr('X', all_master_vars), dtype=np.float64)
            master_solution_array = all_var_values[nontheta_var_indices]

            if previous_master_solution is not None and not continuous_flag:
                if np.array_equal(master_solution_array, previous_master_solution):
                    current_mipgap = master_model.Params.MIPGap
                    master_model.setParam('MIPGap', float(current_mipgap) * 0.8)
            previous_master_solution = master_solution_array

            lower_bound = master_model.ObjVal if continuous_flag else master_model.ObjBound
            best_lower_bound = max(best_lower_bound, lower_bound)
            if lp_phase:
                best_lb_array.append(best_lower_bound)

            if generated_cuts_data['names']:
                if continuous_flag or lp_master_model is None:
                    slacks_array = np.array(master_model.getAttr('Slack', generated_cuts_data['constraints']))
                else:
                    lp_master_model.optimize()
                    slacks_array = np.array(lp_master_model.getAttr('Slack', generated_cuts_data['lp_constraints']))

                scale_array = np.array(generated_cuts_data['scale_factors'], dtype=np.float64)
                abs_unscaled_slacks = np.abs(slacks_array) / scale_array
                threshold = np.percentile(abs_unscaled_slacks, prune_percentile)
                inactive_mask = (abs_unscaled_slacks >= threshold) & (abs_unscaled_slacks > 1)
                cut_inactive_counts = np.where(inactive_mask, cut_inactive_counts + 1, 0)
                prune_mask = inactive_mask & (cut_inactive_counts >= prune_inactive_count)

                if prune_mask.any():
                    prune_indices = np.nonzero(prune_mask)[0]
                    prev_pruned_count = len(pruned_cut_data['names'])
                    constrs_to_remove = []
                    lp_constrs_to_remove = []
                    for idx in prune_indices:
                        for key in pruned_cut_data:
                            pruned_cut_data[key].append(generated_cuts_data[key][idx])
                        constrs_to_remove.append(generated_cuts_data['constraints'][idx])
                        if lp_master_model is not None and generated_cuts_data['lp_constraints'][idx] is not None:
                            lp_constrs_to_remove.append(generated_cuts_data['lp_constraints'][idx])
                    master_model.remove(constrs_to_remove)
                    if lp_constrs_to_remove:
                        lp_master_model.remove(lp_constrs_to_remove)

                    keep_mask = ~prune_mask
                    keep_indices = np.nonzero(keep_mask)[0].tolist()
                    for key in generated_cuts_data:
                        generated_cuts_data[key] = [generated_cuts_data[key][i] for i in keep_indices]
                    cut_inactive_counts = cut_inactive_counts[keep_mask]

                    newly_pruned_var_indices = pruned_cut_data['var_indices'][prev_pruned_count:]
                    newly_pruned_coefs = pruned_cut_data['coefs'][prev_pruned_count:]
                    newly_pruned_constants = pruned_cut_data['constants'][prev_pruned_count:]
                    new_rows, new_constants = build_cut_matrix_rows(newly_pruned_var_indices, newly_pruned_coefs, newly_pruned_constants, num_master_vars)
                    if pruned_cuts_matrix is None:
                        pruned_cuts_matrix = new_rows
                        pruned_cuts_constants_array = new_constants
                    else:
                        pruned_cuts_matrix = sparse_vstack([pruned_cuts_matrix, new_rows], format='csr')
                        pruned_cuts_constants_array = np.concatenate([pruned_cuts_constants_array, new_constants])

            readded_count = 0
            if pruned_cuts_matrix is not None:
                violated_mask = (pruned_cuts_matrix @ all_var_values + pruned_cuts_constants_array) < -1e-6
                if violated_mask.any():
                    violated_indices = np.nonzero(violated_mask)[0]
                    readded_count = len(violated_indices)
                    for idx in violated_indices:
                        var_indices, coefs, constant = pruned_cut_data['var_indices'][idx], pruned_cut_data['coefs'][idx], pruned_cut_data['constants'][idx]
                        expr = LinExpr(coefs.tolist(), all_master_vars_array[var_indices].tolist())
                        expr.addConstant(constant)
                        register_cut(pruned_cut_data['names'][idx], expr, pruned_cut_data['scale_factors'][idx], var_indices, coefs, constant, master_model, lp_master_model, lp_all_vars, generated_cuts_data)
                    cut_inactive_counts = np.concatenate([cut_inactive_counts, np.zeros(readded_count, dtype=np.int32)])
                    keep_mask = ~violated_mask
                    keep_indices = np.nonzero(keep_mask)[0].tolist()
                    for key in pruned_cut_data:
                        pruned_cut_data[key] = [pruned_cut_data[key][i] for i in keep_indices]
                    total_readded += readded_count
                    if pruned_cut_data['names']:
                        pruned_cuts_matrix = pruned_cuts_matrix[keep_mask]
                        pruned_cuts_constants_array = pruned_cuts_constants_array[keep_mask]
                    else:
                        pruned_cuts_matrix = None
                        pruned_cuts_constants_array = None

            subproblem_start_time = time.time()
            futures = {sp_id: executors[sp_id].submit(solve_subproblem, master_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_path_ids}
            subproblem_results = {sp_id: future.result() for sp_id, future in futures.items()}
            subproblem_execution_time = time.time() - subproblem_start_time
            total_subproblem_time += subproblem_execution_time

            subproblem_objectives = {}
            subproblem_cut_intercepts = {}
            subproblem_dual_coefs = {}
            subproblem_feasibility = {}
            for i, sp_id in enumerate(scenario_path_ids):
                obj, cut_intercept, dual_coef, is_feasible = subproblem_results[sp_id]
                subproblem_objectives[sp_id] = obj
                subproblem_cut_intercepts[sp_id] = cut_intercept
                subproblem_dual_coefs[sp_id] = dual_coef
                subproblem_feasibility[sp_id] = is_feasible
                sp_objectives_array[i] = obj

            theta_sum = np.dot(all_var_values[theta_var_indices], scenario_path_prob_array)
            upper_bound = master_model.ObjVal - theta_sum + np.dot(sp_objectives_array, scenario_path_prob_array)
            all_feasible = all(subproblem_feasibility.values())

            if not lp_phase and not continuous_flag and not all_feasible:
                feas_cut_intercepts = {sp_id: v for sp_id, v in subproblem_cut_intercepts.items() if not subproblem_feasibility[sp_id]}
                feas_dual_coefs = {sp_id: v for sp_id, v in subproblem_dual_coefs.items() if not subproblem_feasibility[sp_id]}
                feas_feasibility = {sp_id: False for sp_id in feas_cut_intercepts}
                cut_expressions = add_benders_cuts(feas_cut_intercepts, feas_dual_coefs, feas_feasibility, master_vars_by_sp, theta_vars, master_scaling_stats, master_model_indices, theta_model_indices, all_var_values)
            else:
                cut_expressions = add_benders_cuts(subproblem_cut_intercepts, subproblem_dual_coefs, subproblem_feasibility, master_vars_by_sp, theta_vars, master_scaling_stats, master_model_indices, theta_model_indices, all_var_values)

            for sp_id, (cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant) in cut_expressions.items():
                cut_name = f'OptimalityCut{sp_id}_{iteration}' if subproblem_feasibility.get(sp_id, False) else f'FeasibilityCut{sp_id}_{iteration}'
                register_cut(cut_name, cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant, master_model, lp_master_model, lp_all_vars, generated_cuts_data)

            new_cuts = len(generated_cuts_data['names']) - len(cut_inactive_counts)
            if new_cuts > 0:
                cut_inactive_counts = np.append(cut_inactive_counts, np.zeros(new_cuts, dtype=np.int32))

            optimality_cut_amount = sum(1 for sp_id in cut_expressions if subproblem_feasibility.get(sp_id, False))
            feasibility_cut_amount = len(cut_expressions) - optimality_cut_amount
            optimality_cuts_added += optimality_cut_amount
            feasibility_cuts_added += feasibility_cut_amount

            valid_inequality_derivation_time = 0
            valid_ineq_cut_expressions = None
            if not all_feasible and valid_inequalities_flag and lp_phase:
                valid_inequality_start_time = time.time()
                valid_ineq_cut_expressions = add_valid_inequalities(separation_data, subproblem_feasibility=subproblem_feasibility, all_var_values=all_var_values, master_scaling_stats=master_scaling_stats)
                for cut_name, (cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant) in valid_ineq_cut_expressions.items():
                    register_cut(f'{cut_name}_{iteration}', cut_expression, scale_factor, cut_var_indices, cut_coefs, cut_constant, master_model, lp_master_model, lp_all_vars, generated_cuts_data)
                new_valid_inequalities = len(generated_cuts_data['names']) - len(cut_inactive_counts)
                if new_valid_inequalities > 0:
                    cut_inactive_counts = np.append(cut_inactive_counts, np.zeros(new_valid_inequalities, dtype=np.int32))
                valid_inequality_derivation_time = time.time() - valid_inequality_start_time
                total_valid_inequality_time += valid_inequality_derivation_time
                valid_inequalities_added += len(valid_ineq_cut_expressions)

            if all_feasible and upper_bound < best_upper_bound:
                best_upper_bound = upper_bound
                best_ub_solution = {var.varName: val for var, val, keep in zip(all_master_vars, all_var_values, nontheta_mask) if keep}

            gap = (best_upper_bound - best_lower_bound) / max(1e-6, best_upper_bound)
            log_lines = [
                '-' * 30,
                f"{'LP' if lp_phase else 'MIP'} Iteration {iteration}:",
                f"Upper Bound: {best_upper_bound:.2f}",
                f"Lower Bound: {best_lower_bound:.2f}",
                f"Gap: {(100 * gap):.2f}%",
                f"Number of Optimality Cuts: {optimality_cut_amount}",
                f"Number of Feasibility Cuts: {feasibility_cut_amount}",
                f"Subproblem Execution Time: {subproblem_execution_time:.2f} seconds",
                f"Master Problem Execution Time: {master_execution_time:.2f} seconds"
            ]

            if all_feasible:
                all_ub_found.append([iteration, round(best_lower_bound), round(upper_bound), (100 * gap)])

            if readded_count > 0:
                log_lines.append(f"Readded Cuts from Pool: {readded_count}, Pool Size: {len(pruned_cut_data['names'])}")

            log_file.write('\n'.join(log_lines) + '\n')

            if lp_phase:
                if best_lb_array[0] is None:
                    best_lb_progress = None
                else:
                    best_lb_progress = (best_lower_bound - best_lb_array[0]) / max(abs(best_lower_bound), 1e-10)
                if gap < tolerance or (best_lb_progress is not None and best_lb_progress < 0.001):
                    for var, vtype in original_variable_types.items():
                        var.setAttr('VType', vtype)
                    master_model.update()

                    lp_master_model = master_model.relax()
                    lp_master_model.setParam('OutputFlag', 0)
                    lp_all_vars = np.array(lp_master_model.getVars(), dtype=object)

                    lp_constr_by_name = {c.constrName: c for c in lp_master_model.getConstrs()}
                    for i, name in enumerate(generated_cuts_data['names']):
                        if generated_cuts_data['lp_constraints'][i] is None and name in lp_constr_by_name:
                            generated_cuts_data['lp_constraints'][i] = lp_constr_by_name[name]

                    lp_phase = False
                    lp_phase_iterations = iteration
                    lp_phase_iteration_time = time.time() - total_iteration_time

                    best_upper_bound = float('inf')
                    best_ub_solution = None
            else:
                if gap < tolerance:
                    break

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
        f'Total Feasibility Cuts: {feasibility_cuts_added}',
        f'Total Optimality Cuts: {optimality_cuts_added}',
        f'Total Readded Cuts: {total_readded}',
        f'Subproblem Time: {total_subproblem_time:.2f} seconds',
        f'Master Time: {total_master_time:.2f} seconds',
        f'Iteration Time: {total_iteration_time:.2f} seconds'
    ]

    if valid_inequalities_flag:
        summary_lines.append(f'Valid Inequality Time: {total_valid_inequality_time:.2f} seconds')
        summary_lines.append(f'Number of Valid Inequalities: {valid_inequalities_added}')

    if not callback_flag and original_variable_types:
        summary_lines.append(f'LP Phase Iterations: {lp_phase_iterations}, Iteration Time: {lp_phase_iteration_time}')
        summary_lines.append(f'MIP Phase Iterations: {iteration - lp_phase_iterations}, Iteration Time: {total_iteration_time - lp_phase_iteration_time}')

    summary_lines.append('All upper bounds:')
    summary_lines.append('It.' + '  ' + 'Best LB' + '  ' + 'UB' + '  ' + 'Gap')
    for ub in all_ub_found:
        summary_lines.append(str(ub[0]) + '  ' + str(ub[1]) + '  ' + str(ub[2]) + '  ' + str(ub[3]))

    log_file.write('\n'.join(summary_lines) + '\n')

    if incumbent_solve:
        best_master_solution_array = np.array([best_ub_solution[name] for name in master_var_names], dtype=np.float64)
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, best_master_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}
        for future in futures.values():
            future.result()

        solution_dict = dict(best_ub_solution)
        for _, executor in executors.items():
            future = executor.submit(get_all_subproblem_solution)
            sub_vars = future.result()
            for var_name, var_value in sub_vars.items():
                if var_name not in solution_dict:
                    solution_dict[var_name] = var_value

        if lp_master_model is not None:
            lp_master_model.dispose()
        master_model.dispose()
        master_env.dispose()

        for executor in executors.values():
            executor.shutdown(wait=True)

        return solution_dict

    final_sol_file = os.path.join(results_directory, 'Results.sol')
    with open(final_sol_file, 'w') as f:
        for var_name, var_value in best_ub_solution.items():
            f.write(f"{var_name} {var_value}\n")

    best_master_solution_array = np.array([best_ub_solution[name] for name in master_var_names], dtype=np.float64)

    if not aggregated_subproblems_flag:
        electricity_carry_values, heat_carry_values = write_final_subproblem_solutions(executors, best_master_solution_array, master_dv_indices, results_directory, scenario_paths, numStages, numSubperiods, numSubterms)

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

    else:
        futures = {sp_id: executors[sp_id].submit(solve_subproblem, best_master_solution_array[master_dv_indices[sp_id]]) for sp_id in scenario_paths.keys()}
        for future in futures.values():
            future.result()

        with open(final_sol_file, 'a') as f:
            for executor in executors.values():
                sub_vars = executor.submit(get_all_subproblem_solution).result()
                for var_name, var_value in sub_vars.items():
                    if var_name not in master_var_names:
                        f.write(f"{var_name} {var_value}\n")

    for executor in executors.values():
        executor.shutdown(wait=True)

    #lp_filename = os.path.join(results_directory, f'MasterModel.lp')
    #master_model.write(lp_filename)

    if lp_master_model is not None:
        lp_master_model.dispose()
    master_model.dispose()
    master_env.dispose()