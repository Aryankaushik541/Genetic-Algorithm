"""
Simple Menu for Genetic Algorithm
"""

import sys
import time
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, run_all_functions_parallel, calculate_statistics
from visualization import plot_individual_function, plot_all_functions_combined
from config import NUM_RUNS

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
    
    data = [
        ["Min", f"{stats['min']:.6f}"],
        ["Mean", f"{stats['mean']:.6f}"],
        ["Median", f"{stats['median']:.6f}"],
        ["Std Dev", f"{stats['std']:.6f}"],
        ["Runs", len(results)],
        ["Time", f"{execution_time:.2f}s"]
    ]
    
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
    return stats

def show_all_results(all_results, execution_time):
    print("\n" + "="*80)
    print(" ALL FUNCTIONS (PARALLEL) ".center(80))
    print("="*80)
    
    # Create table
    data = []
    for func_name, results in all_results.items():
        stats = calculate_statistics(results)
        data.append([
            func_name,
            f"{stats['min']:.6f}",
            f"{stats['mean']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['std']:.6f}"
        ])
    
    # Sort by mean
    data.sort(key=lambda x: float(x[2]))
    
    # Add rank
    ranked = []
    for i, row in enumerate(data, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranked.append([emoji] + row)
    
    print(tabulate(ranked,
                   headers=["Rank", "Function", "Min", "Mean", "Median", "Std"],
                   tablefmt="grid"))
    
    print(f"\n{'='*80}")
    print(f"⚡ Time: {execution_time:.2f}s")
    print(f"📊 Functions: {len(all_results)}")
    print(f"🔢 Total runs: {sum(len(r) for r in all_results.values())}")
    print("="*80)

def run_single():
    functions = show_functions()
    
    while True:
        try:
            choice = int(input("\nEnter number (0 to go back): "))
            if choice == 0:
                return
            if 1 <= choice <= len(functions):
                break
            print(f"Enter 1-{len(functions)}")
        except:
            print("Invalid input")
    
    func_name = functions[choice - 1]
    print(f"\n🔄 Running {func_name} ({NUM_RUNS} runs)...")
    
    start = time.time()
    results, history = run_multiple_experiments(func_name, NUM_RUNS)
    exec_time = time.time() - start
    
    stats = show_results(func_name, results, exec_time)
    
    print("\n📊 Generating graph...")
    plot_individual_function(func_name, results, history, stats)
    print("✓ Done!")

def run_all():
    print("\n" + "="*60)
    print(" PARALLEL MODE ".center(60))
    print("="*60)
    print(f"\n⚡ Running all {len(benchmark_functions)} functions in parallel")
    print("Press Enter to start...")
    input()
    
    all_results, plot_data, exec_time = run_all_functions_parallel(num_runs=25)
    
    show_all_results(all_results, exec_time)
    
    print("\n📊 Generating graphs...")
    plot_all_functions_combined(all_results, plot_data)
    print("✓ Done!")

def main():
    while True:
        print_menu()
        
        try:
            choice = int(input("Choice: "))
            
            if choice == 1:
                run_single()
            elif choice == 2:
                run_all()
            elif choice == 3:
                print("\n👋 Bye!\n")
                sys.exit(0)
            else:
                print("Enter 1, 2, or 3")
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!\n")
            sys.exit(0)
        except:
            print("Invalid input")

if __name__ == "__main__":
    main()
