# Benders Decomposition with Partial Non-Anticipativity Relaxation for Multi-Stage Stochastic Clean Energy Transition Planning

*Ahmet Emir Şener, Burak Kocuk, Tuğçe Yüksel*

This repository contains the optimization models, solution algorithm, out-of-sample
simulation, and case-study data used to plan the clean electricity–heat transition of a campus-scale energy system under uncertainty.

## Scope

We study clean energy transition planning for a **campus-scale integrated electricity–heat
system** under both strategic (long-term) and operational (short-term) uncertainty. The
work has three parts:

1. **Multi-stage stochastic model.** A multi-stage stochastic mixed-integer program (MSSP)
   jointly optimizes investment and operational decisions for renewable generation, storage,
   and heat-transfer technologies at high temporal resolution. Technology costs and
   efficiencies evolve stochastically across stages along a scenario tree. A **robust
   reformulation** based on box uncertainty sets protects the plan against unfavorable
   realizations of demand, renewable generation, and heat-pump performance.

2. **Solution algorithm.** Because the extensive form is too large to solve directly, we
   develop a **Benders decomposition algorithm with partial non-anticipativity relaxation**:
   investment variables (and their non-anticipativity constraints) stay in the master
   problem, while operational variables are split into scenario-wise subproblems with their
   non-anticipativity relaxed. Non-anticipativity is restored only at termination via a small
   linear program, with a provably bounded increase in cost. The algorithm is strengthened
   with **valid inequalities** and a **two-phase cut-addition strategy**.

3. **Out-of-sample evaluation.** Nominal and robust investment plans are evaluated through
   **rolling-horizon Monte Carlo simulation**, dispatching each plan against many randomly
   sampled operational realizations (white- and pink-noise deviations) to measure realized
   cost and net-zero reliability.

## Repository Structure

| File | Description |
| --- | --- |
| `main.py` | Runs the full pipeline: data preparation → scenario-tree and model construction → Benders solve → extensive-form verification. |
| `benders.py` | Benders decomposition with partial non-anticipativity relaxation: parallelized scenario-path subproblems, valid-inequality separation, two-phase cut management, and non-anticipativity restoration. |
| `benders_model.py` | Builders for the master problem and scenario-path subproblems. |
| `benders_model_feas.py` | Alternative Benders formulation (feasibility-handling variant). |
| `mssp_model.py` | Extensive-form (deterministic-equivalent) model, used as a benchmark and to verify Benders solutions. |
| `scenario_tree.py` | Constructs the scenario tree from technological-advancement multipliers. |
| `fetch_data.py` | Loads and preprocesses demand, generation, and technology data from `Data/`. |
| `obtain_incumbent.py` | Computes an initial incumbent from a worst-case-advancement instance to warm-start Benders. |
| `robust_simulation.py` | Rolling-horizon Monte Carlo out-of-sample evaluation of investment plans. |
| `plot_convergence.py` | Plots Benders upper/lower bounds and the optimality gap over iterations. |
| `plot_decision_tree.py` | Renders the investment decision tree over the scenario tree. |
| `plot_scenario_path.py` | Visualizes operational dispatch along scenario paths. |
| `Data/` | Excel input files (operational demand, generation profiles, technology specifications). |
| `environment.txt` | Package versions used in the experiments. |

## Requirements

- Python 3.13
- [Gurobi](https://www.gurobi.com/) solver with a valid license
- `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `numba`, `colorednoise`, `gurobipy`

Exact versions used are listed in [`environment.txt`](environment.txt).

## Usage

Run the solver pipeline first — it builds the scenario tree, solves the model with the
proposed algorithm, and writes results (investment plans, logs) to `Results_*` directories:

```bash
python main.py
```

Then evaluate the resulting plans out-of-sample and generate figures:

```bash
python robust_simulation.py   # rolling-horizon Monte Carlo evaluation
python plot_decision_tree.py  # investment decision tree
python plot_convergence.py    # algorithm convergence
```

## Acknowledgements

This work was supported by the Scientific and Technological Research Council of Turkey [grant number 222M243].