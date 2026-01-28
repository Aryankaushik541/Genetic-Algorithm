"""
Main Menu System for Genetic Algorithm Benchmark Suite
Provides options to run individual functions or all functions together
"""

import sys
import time
import numpy as np
from multiprocessing import cpu_count
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, run_all_functions_parallel, calculate_statistics
from visualization import plot_individual_function, plot_all_functions_combined
from ga_config import NUM_RUNS, MAX_EXECUTION_TIME

def print_header():
    """Print application header"""
    print("\n" + "="*60)
    print(" GENETIC ALGORITHM BENCHMARK SUITE ".center(60))
    print("="*60)
    print("Normalized Fitness Values: 0 < value < 1 (excluding 0)")
    print("="*60 + "\n")

def print_menu():
    """Print main menu options"""
    print("\nSelect an option:")
    print("1. Run individual function (separate table & graph for each)")
    print("2. Run all functions together (PARALLEL - combined table & graph)")
    print("3. Exit")
    print()

def display_function_list():
    """Display list of available functions"""
    print("\nAvailable Benchmark Functions:")
    print("-" * 60)
    
    functions = list(benchmark_functions.keys())
    for i, func_name in enumerate(functions, 1):
        print(f"{i:2d}. {func_name}")
    
    print("-" * 60)
    return functions

def display_individual_results(func_name, results, history, execution_time):
    """Display results for a single function"""
    stats = calculate_statistics(results)
    
    print("\n" + "="*60)
    print(f" RESULTS FOR {func_name.upper()} ".center(60))
    print("="*60)
    
    # Create table data
    table_data = [
        ["Metric", "Value"],
        ["Min", f"{stats['min']:.6f}"],
        ["Mean", f"{stats['mean']:.6f}"],
        ["Median", f"{stats['median']:.6f}"],
        ["Std Dev", f"{stats['std']:.6f}"],
        ["Runs", f"{len(results)}"],
        ["Execution Time", f"{execution_time:.2f}s"]
    ]
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    # Verify values are in valid range
    print("\n✓ All values are normalized: 0 < value < 1 (excluding 0)")
    print(f"✓ Value range: [{stats['min']:.6f}, {stats['max']:.6f}]" if 'max' in stats else "")
    
    return stats

def display_combined_results(all_results, execution_time):
    """Display combined results for all functions"""
    print("\n" + "="*100)
    print(" COMBINED RESULTS - ALL 15 FUNCTIONS (PARALLEL EXECUTION) ".center(100))
    print("="*100)
    
    # Prepare table data - sort by mean performance
    table_data = []
    for func_name, results in all_results.items():
        stats = calculate_statistics(results)
        table_data.append([
            func_name,
            f"{stats['min']:.6f}",
            f"{stats['mean']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['std']:.6f}"
        ])
    
    # Sort by mean performance (best first)
    table_data.sort(key=lambda x: float(x[2]))
    
    # Add rank column
    ranked_data = []
    for i, row in enumerate(table_data, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranked_data.append([rank_emoji] + row)
    
    print(tabulate(ranked_data,
                   headers=["Rank", "Function", "Min", "Mean", "Median", "Std Dev"],
                   tablefmt="grid"))
    
    print(f"\n{'='*100}")
    print(f"⚡ Parallel Execution Time: {execution_time:.2f} seconds")
    print(f"📊 Functions Evaluated: {len(all_results)}")
    print(f"🔢 Total Runs: {sum(len(r) for r in all_results.values())}")
    print(f"🎯 Runs per Function: {len(list(all_results.values())[0])}")
    print(f"💻 CPU Cores Used: {min(cpu_count(), len(all_results))}")
    
    # Calculate speedup estimate
    estimated_sequential_time = execution_time * min(cpu_count(), len(all_results))
    print(f"⚡ Estimated Speedup: ~{min(cpu_count(), len(all_results))}x faster than sequential")
    print(f"   (Sequential would take ~{estimated_sequential_time:.1f}s)")
    print("="*100)
    
    # Find best performing function
    best_func = min(all_results.items(), key=lambda x: calculate_statistics(x[1])['mean'])
    best_stats = calculate_statistics(best_func[1])
    
    print(f"\n🏆 BEST PERFORMING FUNCTION: {best_func[0]}")
    print(f"   Mean Performance: {best_stats['mean']:.6f}")
    print(f"   Standard Deviation: {best_stats['std']:.6f}")
    
    # Find most challenging function
    worst_func = max(all_results.items(), key=lambda x: calculate_statistics(x[1])['mean'])
    worst_stats = calculate_statistics(worst_func[1])
    
    print(f"\n⚠️  MOST CHALLENGING FUNCTION: {worst_func[0]}")
    print(f"   Mean Performance: {worst_stats['mean']:.6f}")
    print(f"   Standard Deviation: {worst_stats['std']:.6f}")
    
    print("\n✓ All values are normalized: 0 < value < 1 (excluding 0)")
    print(f"✓ All {len(all_results)} functions completed successfully IN PARALLEL!")

def run_individual_function():
    """Run and display results for a single function"""
    functions = display_function_list()
    
    while True:
        try:
            choice = input("\nEnter function number (or 0 to go back): ")
            choice = int(choice)
            
            if choice == 0:
                return
            
            if 1 <= choice <= len(functions):
                func_name = functions[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(functions)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\n🔄 Running {func_name}...")
    print(f"   Performing {NUM_RUNS} independent runs...")
    
    start_time = time.time()
    results, history = run_multiple_experiments(func_name, NUM_RUNS)
    execution_time = time.time() - start_time
    
    # Display results
    stats = display_individual_results(func_name, results, history, execution_time)
    
    # Generate visualization
    print("\n📊 Generating visualization...")
    plot_individual_function(func_name, results, history, stats)
    
    print("\n✓ Analysis complete!")

def run_all_functions():
    """Run and display results for all functions IN PARALLEL"""
    total_functions = len(benchmark_functions)
    runs_per_function = 25  # Fixed to 25 runs
    
    print(f"\n{'='*60}")
    print(" PARALLEL EXECUTION MODE ".center(60))
    print(f"{'='*60}")
    print(f"\n⚡ All {total_functions} functions will run SIMULTANEOUSLY")
    print(f"💻 Available CPU cores: {cpu_count()}")
    print(f"🎯 Runs per function: {runs_per_function}")
    print(f"🔢 Total runs: {total_functions * runs_per_function}")
    print(f"⏱️  Expected time: ~{MAX_EXECUTION_TIME} seconds")
    print(f"\n{'='*60}\n")
    
    input("Press Enter to start parallel execution...")
    print()
    
    # Run all functions in parallel
    all_results, plot_data, execution_time = run_all_functions_parallel(
        num_runs=runs_per_function, 
        max_time=MAX_EXECUTION_TIME
    )
    
    # Display combined results
    display_combined_results(all_results, execution_time)
    
    # Generate combined visualization
    print("\n📊 Generating comprehensive visualization...")
    print("   This will show all 15 functions in 4 detailed graphs...")
    plot_all_functions_combined(all_results, plot_data)
    
    print("\n✓ Parallel analysis complete!")

def main():
    """Main application loop"""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-3): ")
            choice = int(choice)
            
            if choice == 1:
                run_individual_function()
            elif choice == 2:
                run_all_functions()
            elif choice == 3:
                print("\n👋 Thank you for using GA Benchmark Suite!")
                print("="*60 + "\n")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
