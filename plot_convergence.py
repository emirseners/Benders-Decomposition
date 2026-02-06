import matplotlib.pyplot as plt
import os

def parse_benders_log(log_file_path):
    upper_bounds = []
    lower_bounds = []
    gaps = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if line.startswith('Upper Bound:'):
                value = float(line.split(':')[1].strip())
                upper_bounds.append(value)
            elif line.startswith('Lower Bound:'):
                value = float(line.split(':')[1].strip())
                lower_bounds.append(value)
            elif line.startswith('Gap:'):
                value = float(line.split(':')[1].strip().replace('%', ''))
                gaps.append(value)
    
    return upper_bounds, lower_bounds, gaps

def plot_convergence(upper_bounds, lower_bounds, gaps, output_path=None):
    iterations = range(1, len(upper_bounds) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(iterations, upper_bounds, 'r-o', label='Upper Bound', markersize=4)
    ax1.plot(iterations, lower_bounds, 'b-s', label='Lower Bound', markersize=4)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Objective Value', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(iterations, gaps, 'g-^', label='Gap (%)', markersize=4)
    ax2.set_ylabel('Gap (%)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    plt.title('Benders Decomposition Convergence', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

def main():
    numStages = 3
    numSubperiods = 5
    numSubterms = 1092
    results_directory = f'Results_{numStages}_{numSubperiods}_{numSubterms}'
    
    log_file_path = os.path.join(results_directory, 'BendersLog.txt')
    
    upper_bounds, lower_bounds, gaps = parse_benders_log(log_file_path)
    output_path = os.path.join(results_directory, 'convergence_plot.png')
    plot_convergence(upper_bounds, lower_bounds, gaps, output_path=output_path)

if __name__ == '__main__':
    main()