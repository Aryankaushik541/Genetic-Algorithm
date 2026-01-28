

from main_experiment import run_experiment
from results_table import display_results_table, get_best_function_info
from visualization import create_all_graphs

def main():
    print("="*60)
    print(" GENETIC ALGORITHM BENCHMARK ANALYSIS ".center(60))
    print("="*60)
    
    # Run the complete experiment
    results, plot_data, all_scores, exec_time = run_experiment(num_runs=30)
    
    # Display results table
    display_results_table(results, exec_time)
    
    # Get best function info
    best_function, best_idx = get_best_function_info(results)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    create_all_graphs(results, plot_data, all_scores, best_function)
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE! ".center(60))
    print("="*60)

if __name__ == "__main__":
    main()