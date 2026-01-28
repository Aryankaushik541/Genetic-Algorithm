"""
Quick Test Script
Fast testing of individual functions without full menu system
"""

import sys
import time
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, calculate_statistics

def quick_test(function_name, num_runs=10, verbose=True):
    """
    Quick test of a single function
    
    Args:
        function_name: Name of benchmark function
        num_runs: Number of runs (default: 10)
        verbose: Print detailed output
    
    Returns:
        Statistics dictionary
    """
    if function_name not in benchmark_functions:
        print(f"❌ Error: Function '{function_name}' not found!")
        print(f"Available functions: {', '.join(benchmark_functions.keys())}")
        return None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" QUICK TEST: {function_name.upper()} ".center(60))
        print(f"{'='*60}\n")
        print(f"Running {num_runs} independent experiments...")
    
    start_time = time.time()
    results, history = run_multiple_experiments(function_name, num_runs)
    exec_time = time.time() - start_time
    
    stats = calculate_statistics(results)
    
    if verbose:
        print(f"\n✓ Completed in {exec_time:.2f} seconds\n")
        
        # Display results
        table_data = [
            ["Metric", "Value"],
            ["Best (Min)", f"{stats['min']:.6f}"],
            ["Average (Mean)", f"{stats['mean']:.6f}"],
            ["Median", f"{stats['median']:.6f}"],
            ["Std Deviation", f"{stats['std']:.6f}"],
            ["Worst (Max)", f"{stats['max']:.6f}"],
            ["Runs", f"{num_runs}"],
            ["Time", f"{exec_time:.2f}s"]
        ]
        
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        
        # Performance assessment
        print(f"\n{'='*60}")
        if stats['mean'] < 0.1:
            print("🏆 Performance: EXCELLENT")
        elif stats['mean'] < 0.3:
            print("⭐ Performance: VERY GOOD")
        elif stats['mean'] < 0.5:
            print("✓ Performance: GOOD")
        else:
            print("⚠ Performance: MODERATE")
        
        if stats['std'] < 0.01:
            print("🎯 Consistency: VERY HIGH")
        elif stats['std'] < 0.05:
            print("✓ Consistency: HIGH")
        else:
            print("⚠ Consistency: MODERATE")
        print(f"{'='*60}\n")
    
    return stats

def compare_functions(function_names, num_runs=10):
    """
    Quick comparison of multiple functions
    
    Args:
        function_names: List of function names
        num_runs: Number of runs per function
    """
    print(f"\n{'='*80}")
    print(f" COMPARING {len(function_names)} FUNCTIONS ".center(80))
    print(f"{'='*80}\n")
    
    results = []
    
    for func_name in function_names:
        print(f"Testing {func_name}...", end=" ", flush=True)
        stats = quick_test(func_name, num_runs, verbose=False)
        if stats:
            results.append({
                'function': func_name,
                'mean': stats['mean'],
                'std': stats['std']
            })
            print(f"✓ Mean: {stats['mean']:.6f}")
    
    # Display comparison
    print(f"\n{'='*80}")
    print(" COMPARISON RESULTS ".center(80))
    print(f"{'='*80}\n")
    
    # Sort by mean performance
    results.sort(key=lambda x: x['mean'])
    
    table_data = []
    for i, r in enumerate(results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        table_data.append([
            rank_emoji,
            r['function'],
            f"{r['mean']:.6f}",
            f"{r['std']:.6f}"
        ])
    
    print(tabulate(table_data,
                   headers=['Rank', 'Function', 'Mean', 'Std Dev'],
                   tablefmt='grid'))
    
    print(f"\n🏆 Best: {results[0]['function']} (Mean: {results[0]['mean']:.6f})")
    print(f"{'='*80}\n")

def list_functions():
    """List all available benchmark functions"""
    print(f"\n{'='*60}")
    print(" AVAILABLE BENCHMARK FUNCTIONS ".center(60))
    print(f"{'='*60}\n")
    
    functions = list(benchmark_functions.keys())
    
    # Split into categories
    unimodal = ['Sphere', 'Zakharov', 'Schwefel_222', 'Schwefel_12', 
                'Sum_Diff_Powers', 'Matyas', 'Dixon_Price', 
                'Rotated_Hyper_Ellipsoid', 'Bent_Cigar', 'Booth']
    multimodal = ['Rastrigin', 'Ackley', 'Griewank', 'Levy', 'Perm']
    
    print("UNIMODAL FUNCTIONS (Single Global Minimum):")
    for i, func in enumerate(unimodal, 1):
        if func in functions:
            print(f"  {i:2d}. {func}")
    
    print("\nMULTIMODAL FUNCTIONS (Multiple Local Minima):")
    for i, func in enumerate(multimodal, 1):
        if func in functions:
            print(f"  {i:2d}. {func}")
    
    print(f"\n{'='*60}\n")

# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show help
        print("\n" + "="*60)
        print(" QUICK TEST UTILITY ".center(60))
        print("="*60)
        print("\nUsage:")
        print("  python quick_test.py <function_name> [num_runs]")
        print("  python quick_test.py list")
        print("  python quick_test.py compare <func1> <func2> ...")
        print("\nExamples:")
        print("  python quick_test.py Sphere")
        print("  python quick_test.py Rastrigin 20")
        print("  python quick_test.py list")
        print("  python quick_test.py compare Sphere Rastrigin Ackley")
        print("\n" + "="*60 + "\n")
        
    elif sys.argv[1] == "list":
        # List all functions
        list_functions()
        
    elif sys.argv[1] == "compare":
        # Compare multiple functions
        if len(sys.argv) < 3:
            print("❌ Error: Please specify at least one function to compare")
            print("Example: python quick_test.py compare Sphere Rastrigin")
        else:
            function_names = sys.argv[2:]
            num_runs = 10
            compare_functions(function_names, num_runs)
            
    else:
        # Test single function
        function_name = sys.argv[1]
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        quick_test(function_name, num_runs)
