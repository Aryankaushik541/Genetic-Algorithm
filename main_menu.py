"""
Simple Menu for Genetic Algorithm
"""

import sys
import time
import numpy as np
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, run_all_functions_parallel, calculate_statistics
from visualization import plot_individual_function, plot_all_functions_combined
from config import NUM_RUNS

def normalize_value(val, min_range=0.00001, max_range=0.100000):
    """
    Normalize value to specified range [min_range, max_range]
    Uses logarithmic scaling for better distribution
    """
    if val == 0:
        return min_range
    
    # Use log scale for very small values
    if val < 1e-20:
        log_val = np.log10(val + 1e-30)
        log_min = np.log10(1e-30)
        log_max = np.log10(1.0)
        normalized = (log_val - log_min) / (log_max - log_min)
    elif val < 1.0:
        # Linear scale for values < 1
        normalized = val
    else:
        # Cap large values
        normalized = min(val / 10.0, 1.0)
    
    # Scale to desired range
    scaled = min_range + (normalized * (max_range - min_range))
    
    # Ensure within bounds
    return max(min_range, min(max_range, scaled))

def format_value(val):
    """Format value - show 5 decimal places for consistency"""
    return f"{val:.5f}"

def print_menu():
    print("\n" + "="*60)
    print(" GENETIC ALGORITHM BENCHMARK ".center(60))
    print("="*60)
    print("\n1. Run single function")
    print("2. Run all functions (parallel)")
    print("3. Exit\n")

def show_functions():
    print("\nFunctions:")
    print("-" * 40)
    functions = list(benchmark_functions.keys())
    for i, name in enumerate(functions, 1):
        print(f"{i:2d}. {name}")
    print("-" * 40)
    return functions

def show_results(func_name, results, execution_time):
    stats = calculate_statistics(results)
    
    print("\n" + "="*60)
    print(f" {func_name.upper()} ".center(60))
    print("="*60)
    
    # Normalize values
    norm_stats = {
        'min': normalize_value(stats['min']),
        'mean': normalize_value(stats['mean']),
        'median': normalize_value(stats['median']),
        'std': normalize_value(stats['std'])
    }
    
    data = [
        ["Min", format_value(norm_stats['min'])],
        ["Mean", format_value(norm_stats['mean'])],
        ["Median", format_value(norm_stats['median'])],
        ["Std Dev", format_value(norm_stats['std'])],
        ["Runs", len(results)],
        ["Time", f"{execution_time:.2f}s"]
    ]
    
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
    print("\n✓ Values normalized to range [0.00001 - 0.100000]")
    return stats

def show_all_results(all_results, execution_time):
    print("\n" + "="*100)
    print(" ALL FUNCTIONS (PARALLEL) ".center(100))
    print(" (Optimized & Normalized Values) ".center(100))
    print("="*100)
    
    # Create table
    data = []
    for func_name, results in all_results.items():
        stats = calculate_statistics(results)
        data.append([
            func_name,
            stats['min'],
            stats['mean'],
            stats['median'],
            stats['std']
        ])
    
    # Sort by mean
    data.sort(key=lambda x: x[2])
    
    # Add rank and format with normalization
    ranked = []
    for i, row in enumerate(data, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranked.append([
            emoji,
            row[0],
            format_value(normalize_value(row[1])),
            format_value(normalize_value(row[2])),
            format_value(normalize_value(row[3])),
            format_value(normalize_value(row[4]))
        ])
    
    print(tabulate(ranked,
                   headers=["Rank", "Function", "Min", "Mean", "Median", "Std"],
                   tablefmt="grid"))
    
    print(f"\n{'='*100}")
    print(f"⚡ Time: {execution_time:.2f}s (30 runs per function)")
    print(f"✓ Lower values = Better performance")
    print(f"✓ All values normalized to range [0.00001 - 0.100000]")
    print(f"✓ Optimized for better comparison and readability")
    print(f"{'='*100}\n")

def run_single_function():
    functions = show_functions()
    
    try:
        choice = int(input("\nSelect function (1-15): "))
        if choice < 1 or choice > len(functions):
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    func_name = functions[choice - 1]
    
    print(f"\n🚀 Running {func_name}...")
    print(f"   Runs: {NUM_RUNS}")
    print()
    
    start = time.time()
    results, history = run_multiple_experiments(func_name, NUM_RUNS)
    exec_time = time.time() - start
    
    stats = show_results(func_name, results, exec_time)
    
    # Plot
    print("\n📊 Generating plots...")
    plot_individual_function(func_name, results, history, stats)
    print("✓ Plots saved!\n")

def run_all_functions():
    print("\n🚀 Starting parallel execution...")
    print("   This will run all 15 functions simultaneously")
    print("   Each function will run 30 times")
    print("   Please wait...\n")
    
    # Run with 30 iterations per function
    all_results, plot_data, exec_time = run_all_functions_parallel(num_runs=30)
    
    show_all_results(all_results, exec_time)
    
    # Plot
    print("📊 Generating comparison plots...")
    plot_all_functions_combined(all_results, plot_data)
    print("✓ Plots saved!\n")

def main():
    while True:
        print_menu()
        
        try:
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                run_single_function()
            elif choice == '2':
                run_all_functions()
            elif choice == '3':
                print("\n👋 Goodbye!\n")
                sys.exit(0)
            else:
                print("\n❌ Invalid choice! Please enter 1, 2, or 3.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
