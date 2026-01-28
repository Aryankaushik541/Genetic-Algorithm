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

def format_value(val):
    """Format value - show full decimal precision with all zeros visible"""
    if val == 0:
        return "0"
    
    # Convert to string to check magnitude
    str_val = f"{val:.2e}"  # Get scientific notation to determine magnitude
    
    # Extract exponent
    if 'e' in str_val:
        mantissa, exponent = str_val.split('e')
        exp_val = int(exponent)
        
        # For very small numbers (negative exponent), show full decimal
        if exp_val < 0:
            # Calculate number of decimal places needed
            decimal_places = abs(exp_val) + 15  # Extra precision
            formatted = f"{val:.{decimal_places}f}"
            # Remove trailing zeros but keep significant digits
            formatted = formatted.rstrip('0').rstrip('.')
            return formatted
        else:
            # For larger numbers, use reasonable precision
            if val < 1:
                return f"{val:.15f}".rstrip('0').rstrip('.')
            else:
                return f"{val:.6f}".rstrip('0').rstrip('.')
    
    return f"{val:.15f}".rstrip('0').rstrip('.')

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
        ["Min", format_value(stats['min'])],
        ["Mean", format_value(stats['mean'])],
        ["Median", format_value(stats['median'])],
        ["Std Dev", format_value(stats['std'])],
        ["Runs", len(results)],
        ["Time", f"{execution_time:.2f}s"]
    ]
    
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
    return stats

def show_all_results(all_results, execution_time):
    print("\n" + "="*100)
    print(" ALL FUNCTIONS (PARALLEL) ".center(100))
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
    
    # Add rank and format
    ranked = []
    for i, row in enumerate(data, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranked.append([
            emoji,
            row[0],
            format_value(row[1]),
            format_value(row[2]),
            format_value(row[3]),
            format_value(row[4])
        ])
    
    print(tabulate(ranked,
                   headers=["Rank", "Function", "Min", "Mean", "Median", "Std"],
                   tablefmt="grid"))
    
    print(f"\n{'='*100}")
    print(f"⚡ Time: {execution_time:.2f}s")
    print(f"✓ Lower values = Better performance")
    print(f"✓ Values shown in full decimal format (all zeros visible)")
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
    print("   Please wait...\n")
    
    all_results, plot_data, exec_time = run_all_functions_parallel()
    
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
