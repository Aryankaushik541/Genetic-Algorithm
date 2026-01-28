"""
Quick Test Script
Fast testing of individual functions without full menu system
"""

import sys
import time
from multiprocessing import Pool, cpu_count
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, calculate_statistics
from ga_config import EXCELLENT_THRESHOLD, GOOD_THRESHOLD, MODERATE_THRESHOLD, VERY_CONSISTENT, CONSISTENT

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
        if stats['mean'] < EXCELLENT_THRESHOLD:
            print("🏆 Performance: EXCELLENT")
        elif stats['mean'] < GOOD_THRESHOLD:
            print("⭐ Performance: VERY GOOD")
        elif stats['mean'] < MODERATE_THRESHOLD:
            print("✓ Performance: GOOD")
        else:
            print("⚠ Performance: MODERATE")
        
        if stats['std'] < VERY_CONSISTENT:
            print("🎯 Consistency: VERY HIGH")
        elif stats['std'] < CONSISTENT:
            print("✓ Consistency: HIGH")
        else:
            print("⚠ Consistency: MODERATE")
        print(f"{'='*60}\n")
    
    return stats

def _worker_test_function(args):
    func_name, num_runs = args
    try:
        results, _ = run_multiple_experiments(func_name, num_runs)
        stats = calculate_statistics(results)
        return {
            'function': func_name,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'median': stats['median'],
            'success': True
        }
    except Exception as e:
        return {
            'function': func_name,
            'error': str(e),
            'success': False
        }

def compare_functions(function_names, num_runs=10, parallel=True):
    """
    Quick comparison of multiple functions
    
    Args:
        function_names: List of function names
        num_runs: Number of runs per function
        parallel: If True, run functions in parallel (default: True)
    """
    print(f"\n{'='*80}")
    print(f" COMPARING {len(function_names)} FUNCTIONS ".center(80))
    print(f"{'='*80}\n")
    
    valid_functions = [f for f in function_names if f in benchmark_functions]
    invalid_functions = [f for f in function_names if f not in benchmark_functions]
    
    if invalid_functions:
        print(f"⚠ Skipping unknown functions: {', '.join(invalid_functions)}")
    
    if not valid_functions:
        print("❌ No valid functions to test!")
        return
    
    start_time = time.time()
    results = []
    
    if parallel and len(valid_functions) > 1:
        num_workers = min(cpu_count(), len(valid_functions))
        print(f"⚡ Running {len(valid_functions)} functions in PARALLEL")
        print(f"   Using {num_workers} CPU cores")
        print(f"   {num_runs} runs per function")
        print()
        
        worker_args = [(func_name, num_runs) for func_name in valid_functions]
        
        with Pool(processes=num_workers) as pool:
            parallel_results = pool.map(_worker_test_function, worker_args)
        
        for r in parallel_results:
            if r['success']:
                results.append(r)
                print(f"✓ {r['function']}: Mean = {r['mean']:.6f}")
            else:
                print(f"❌ {r['function']}: Error - {r['error']}")
    else:
        print(f"Running {len(valid_functions)} functions sequentially...")
        print()
        
        for func_name in valid_functions:
            print(f"Testing {func_name}...", end=" ", flush=True)
            stats = quick_test(func_name, num_runs, verbose=False)
            if stats:
                results.append({
                    'function': func_name,
                    'mean': stats['mean'],
                    'std': stats['std']
                })
                print(f"✓ Mean: {stats['mean']:.6f}")
    
    exec_time = time.time() - start_time
    
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
    print(f"⏱ Total time: {exec_time:.2f} seconds")
    if parallel and len(valid_functions) > 1:
        print(f"⚡ Speedup: ~{len(valid_functions)}x faster than sequential")
    print(f"{'='*80}\n")

def test_all_parallel(num_runs=10):
    all_functions = list(benchmark_functions.keys())
    total_functions = len(all_functions)
    
    print(f"\n{'='*80}")
    print(" PARALLEL TEST: ALL BENCHMARK FUNCTIONS ".center(80))
    print(f"{'='*80}\n")
    
    print(f"🚀 Testing all {total_functions} functions IN PARALLEL")
    print(f"   CPU cores available: {cpu_count()}")
    print(f"   Workers to use: {min(cpu_count(), total_functions)}")
    print(f"   Runs per function: {num_runs}")
    print(f"   Total GA runs: {total_functions * num_runs}")
    print()
    
    start_time = time.time()
    
    worker_args = [(func_name, num_runs) for func_name in all_functions]
    num_workers = min(cpu_count(), total_functions)
    
    print("⚡ Starting parallel execution...")
    print()
    
    results = []
    
    with Pool(processes=num_workers) as pool:
        parallel_results = pool.map(_worker_test_function, worker_args)
    
    for r in parallel_results:
        if r['success']:
            results.append(r)
            print(f"✓ {r['function']:25s} Mean: {r['mean']:.6f}  Std: {r['std']:.6f}")
        else:
            print(f"❌ {r['function']:25s} Error: {r['error']}")
    
    exec_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(" RESULTS SUMMARY ".center(80))
    print(f"{'='*80}\n")
    
    results.sort(key=lambda x: x['mean'])
    
    excellent = [r for r in results if r['mean'] < EXCELLENT_THRESHOLD]
    good = [r for r in results if EXCELLENT_THRESHOLD <= r['mean'] < GOOD_THRESHOLD]
    moderate = [r for r in results if GOOD_THRESHOLD <= r['mean'] < MODERATE_THRESHOLD]
    poor = [r for r in results if r['mean'] >= MODERATE_THRESHOLD]
    
    table_data = []
    for i, r in enumerate(results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        perf_emoji = "🏆" if r['mean'] < EXCELLENT_THRESHOLD else "⭐" if r['mean'] < GOOD_THRESHOLD else "✓" if r['mean'] < MODERATE_THRESHOLD else "⚠"
        table_data.append([
            rank_emoji,
            r['function'],
            f"{r['mean']:.6f}",
            f"{r['std']:.6f}",
            f"{r['min']:.6f}",
            f"{r['max']:.6f}",
            perf_emoji
        ])
    
    print(tabulate(table_data,
                   headers=['Rank', 'Function', 'Mean', 'Std Dev', 'Best', 'Worst', 'Grade'],
                   tablefmt='grid'))
    
    print(f"\n📊 PERFORMANCE BREAKDOWN:")
    print(f"   🏆 Excellent (< {EXCELLENT_THRESHOLD}): {len(excellent)} functions")
    print(f"   ⭐ Good (< {GOOD_THRESHOLD}):      {len(good)} functions")
    print(f"   ✓  Moderate (< {MODERATE_THRESHOLD}):  {len(moderate)} functions")
    print(f"   ⚠  Poor (≥ {MODERATE_THRESHOLD}):     {len(poor)} functions")
    
    print(f"\n🏆 Best performer: {results[0]['function']} (Mean: {results[0]['mean']:.6f})")
    print(f"⏱ Total execution time: {exec_time:.2f} seconds")
    print(f"⚡ Speedup: ~{total_functions}x faster than sequential")
    print(f"{'='*80}\n")
    
    return results

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
        print("  python quick_test.py all [num_runs]")
        print("\nExamples:")
        print("  python quick_test.py Sphere")
        print("  python quick_test.py Rastrigin 20")
        print("  python quick_test.py list")
        print("  python quick_test.py compare Sphere Rastrigin Ackley")
        print("  python quick_test.py all")
        print("  python quick_test.py all 20")
        print("\n" + "="*60 + "\n")
        
    elif sys.argv[1] == "list":
        # List all functions
        list_functions()
        
    elif sys.argv[1] == "all":
        # Test all functions in parallel
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        test_all_parallel(num_runs)
        
    elif sys.argv[1] == "compare":
        # Compare multiple functions
        if len(sys.argv) < 3:
            print("❌ Error: Please specify at least one function to compare")
            print("Example: python quick_test.py compare Sphere Rastrigin")
        else:
            function_names = sys.argv[2:]
            num_runs = 10
            compare_functions(function_names, num_runs, parallel=True)
            
    else:
        # Test single function
        function_name = sys.argv[1]
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        quick_test(function_name, num_runs)
