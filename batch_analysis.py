"""
Batch Analysis Utility
Automated testing and comparison of different GA configurations
"""

import numpy as np
import time
from tabulate import tabulate
from ga_algorithm import run_multiple_experiments, calculate_statistics
from benchmark_functions import benchmark_functions

def compare_configurations(function_name, configs, num_runs=10):
    """
    Compare different GA configurations on a single function
    
    Args:
        function_name: Name of benchmark function
        configs: List of configuration dictionaries
        num_runs: Number of runs per configuration
    
    Returns:
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f" COMPARING CONFIGURATIONS ON {function_name.upper()} ".center(80))
    print(f"{'='*80}\n")
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"Testing Configuration {i}/{len(configs)}...")
        print(f"  Parameters: {config}")
        
        # Temporarily modify config (in real implementation, pass to GA)
        start_time = time.time()
        fitness_values, _ = run_multiple_experiments(function_name, num_runs)
        exec_time = time.time() - start_time
        
        stats = calculate_statistics(fitness_values)
        
        results.append({
            'config_id': i,
            'config': str(config),
            'min': stats['min'],
            'mean': stats['mean'],
            'median': stats['median'],
            'std': stats['std'],
            'time': exec_time
        })
        
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Time: {exec_time:.2f}s\n")
    
    # Display comparison table
    table_data = []
    for r in results:
        table_data.append([
            f"Config {r['config_id']}",
            f"{r['min']:.6f}",
            f"{r['mean']:.6f}",
            f"{r['median']:.6f}",
            f"{r['std']:.6f}",
            f"{r['time']:.2f}s"
        ])
    
    print(tabulate(table_data,
                   headers=['Configuration', 'Min', 'Mean', 'Median', 'Std Dev', 'Time'],
                   tablefmt='grid'))
    
    # Find best configuration
    best_idx = np.argmin([r['mean'] for r in results])
    print(f"\n🏆 Best Configuration: Config {results[best_idx]['config_id']}")
    print(f"   Mean: {results[best_idx]['mean']:.6f}")
    print(f"   Std Dev: {results[best_idx]['std']:.6f}")
    
    return results

def batch_test_functions(function_names=None, num_runs=5):
    """
    Quick batch test of multiple functions
    
    Args:
        function_names: List of function names (None = all functions)
        num_runs: Number of runs per function
    """
    if function_names is None:
        function_names = list(benchmark_functions.keys())
    
    print(f"\n{'='*80}")
    print(f" BATCH TESTING {len(function_names)} FUNCTIONS ".center(80))
    print(f"{'='*80}\n")
    
    results = []
    total_start = time.time()
    
    for func_name in function_names:
        print(f"Testing {func_name}...", end=" ")
        start = time.time()
        
        fitness_values, _ = run_multiple_experiments(func_name, num_runs)
        stats = calculate_statistics(fitness_values)
        exec_time = time.time() - start
        
        results.append({
            'function': func_name,
            'min': stats['min'],
            'mean': stats['mean'],
            'std': stats['std'],
            'time': exec_time
        })
        
        print(f"Done ({exec_time:.2f}s)")
    
    total_time = time.time() - total_start
    
    # Display results
    print(f"\n{'='*80}")
    print(" BATCH TEST RESULTS ".center(80))
    print(f"{'='*80}\n")
    
    table_data = []
    for r in results:
        table_data.append([
            r['function'],
            f"{r['min']:.6f}",
            f"{r['mean']:.6f}",
            f"{r['std']:.6f}",
            f"{r['time']:.2f}s"
        ])
    
    print(tabulate(table_data,
                   headers=['Function', 'Min', 'Mean', 'Std Dev', 'Time'],
                   tablefmt='grid'))
    
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Average Time per Function: {total_time/len(function_names):.2f} seconds")
    
    return results

def export_results_csv(results, filename='ga_results.csv'):
    """
    Export results to CSV file
    
    Args:
        results: Results dictionary from batch analysis
        filename: Output CSV filename
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        if isinstance(results, list) and len(results) > 0:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n✓ Results exported to {filename}")

def sensitivity_analysis(function_name, parameter_name, parameter_values, num_runs=10):
    """
    Analyze sensitivity to a specific parameter
    
    Args:
        function_name: Name of benchmark function
        parameter_name: Parameter to vary (e.g., 'mutation_rate')
        parameter_values: List of values to test
        num_runs: Number of runs per value
    """
    print(f"\n{'='*80}")
    print(f" SENSITIVITY ANALYSIS: {parameter_name.upper()} ".center(80))
    print(f" Function: {function_name} ".center(80))
    print(f"{'='*80}\n")
    
    results = []
    
    for value in parameter_values:
        print(f"Testing {parameter_name} = {value}...", end=" ")
        
        # In real implementation, modify GA parameter
        fitness_values, _ = run_multiple_experiments(function_name, num_runs)
        stats = calculate_statistics(fitness_values)
        
        results.append({
            'parameter_value': value,
            'mean': stats['mean'],
            'std': stats['std']
        })
        
        print(f"Mean: {stats['mean']:.6f}")
    
    # Display results
    print(f"\n{'='*80}")
    print(" SENSITIVITY RESULTS ".center(80))
    print(f"{'='*80}\n")
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['parameter_value']}",
            f"{r['mean']:.6f}",
            f"{r['std']:.6f}"
        ])
    
    print(tabulate(table_data,
                   headers=[parameter_name, 'Mean Fitness', 'Std Dev'],
                   tablefmt='grid'))
    
    # Find optimal value
    best_idx = np.argmin([r['mean'] for r in results])
    print(f"\n🏆 Optimal {parameter_name}: {results[best_idx]['parameter_value']}")
    print(f"   Mean Fitness: {results[best_idx]['mean']:.6f}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("Batch Analysis Utility")
    print("=" * 80)
    print("\nAvailable Functions:")
    print("1. compare_configurations() - Compare different GA settings")
    print("2. batch_test_functions() - Quick test of multiple functions")
    print("3. export_results_csv() - Export results to CSV")
    print("4. sensitivity_analysis() - Parameter sensitivity analysis")
    
    print("\n\nExample: Quick batch test of 3 functions")
    print("-" * 80)
    
    # Quick test
    test_functions = ['Sphere', 'Rastrigin', 'Ackley']
    results = batch_test_functions(test_functions, num_runs=5)
    
    print("\n✓ Batch analysis complete!")
