"""
GA Utilities - Simple unified utility script
Combines quick testing, batch analysis, and export functionality
"""

import json
import csv
import time
from datetime import datetime
from ga_algorithm import run_multiple_experiments, calculate_statistics
from benchmark_functions import benchmark_functions

# ============================================================
# QUICK TEST - Fast function testing
# ============================================================

def quick_test(func_name, num_runs=10):
    """Quick test a single function"""
    print(f"\n🔬 Quick Test: {func_name}")
    print("="*60)
    print(f"Runs: {num_runs}\n")
    
    start = time.time()
    results, _ = run_multiple_experiments(func_name, num_runs)
    exec_time = time.time() - start
    
    stats = calculate_statistics(results)
    
    print(f"✓ Completed in {exec_time:.2f}s\n")
    print(f"Min:    {stats['min']:.6f}")
    print(f"Mean:   {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Std:    {stats['std']:.6f}")
    print(f"Max:    {stats['max']:.6f}")
    print("="*60 + "\n")
    
    return results

# ============================================================
# BATCH TEST - Test multiple functions
# ============================================================

def batch_test(func_names, num_runs=10):
    """Test multiple functions and compare"""
    print(f"\n📊 Batch Test: {len(func_names)} functions")
    print("="*60)
    print(f"Runs per function: {num_runs}\n")
    
    all_results = {}
    start = time.time()
    
    for i, func_name in enumerate(func_names, 1):
        print(f"[{i}/{len(func_names)}] {func_name}...", end=" ", flush=True)
        results, _ = run_multiple_experiments(func_name, num_runs)
        all_results[func_name] = results
        print("✓")
    
    exec_time = time.time() - start
    print(f"\n✓ All completed in {exec_time:.2f}s\n")
    
    # Show comparison
    print("Comparison:")
    print("-"*60)
    sorted_funcs = sorted(all_results.items(), 
                         key=lambda x: calculate_statistics(x[1])['mean'])
    
    for i, (func_name, results) in enumerate(sorted_funcs, 1):
        stats = calculate_statistics(results)
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{rank} {func_name:20s} Mean: {stats['mean']:.6f}")
    
    print("="*60 + "\n")
    return all_results

# ============================================================
# EXPORT - Export results to files
# ============================================================

def export_csv(results_dict, filename):
    """Export to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Function', 'Min', 'Mean', 'Median', 'Std', 'Max', 'Runs'])
        
        for func_name, results in results_dict.items():
            stats = calculate_statistics(results)
            writer.writerow([
                func_name,
                f"{stats['min']:.6f}",
                f"{stats['mean']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['max']:.6f}",
                len(results)
            ])
    print(f"✓ CSV: {filename}")

def export_json(results_dict, filename):
    """Export to JSON"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_functions': len(results_dict),
        'results': {}
    }
    
    for func_name, results in results_dict.items():
        stats = calculate_statistics(results)
        data['results'][func_name] = {
            'min': float(stats['min']),
            'mean': float(stats['mean']),
            'median': float(stats['median']),
            'std': float(stats['std']),
            'max': float(stats['max']),
            'values': [float(r) for r in results],
            'runs': len(results)
        }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ JSON: {filename}")

def export_markdown(results_dict, filename):
    """Export to Markdown"""
    with open(filename, 'w') as f:
        f.write("# GA Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Functions:** {len(results_dict)}\n\n")
        
        f.write("| Rank | Function | Min | Mean | Median | Std |\n")
        f.write("|------|----------|-----|------|--------|-----|\n")
        
        sorted_results = sorted(results_dict.items(), 
                               key=lambda x: calculate_statistics(x[1])['mean'])
        
        for i, (func_name, results) in enumerate(sorted_results, 1):
            stats = calculate_statistics(results)
            rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            f.write(f"| {rank} | {func_name} | {stats['min']:.6f} | "
                   f"{stats['mean']:.6f} | {stats['median']:.6f} | {stats['std']:.6f} |\n")
    
    print(f"✓ Markdown: {filename}")

def export_results(results_dict, format='all', base_name='ga_results'):
    """Export results to file(s)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"{base_name}_{timestamp}"
    
    print("\n📤 Exporting results...")
    
    if format in ['all', 'csv']:
        export_csv(results_dict, f"{base}.csv")
    if format in ['all', 'json']:
        export_json(results_dict, f"{base}.json")
    if format in ['all', 'md']:
        export_markdown(results_dict, f"{base}.md")
    
    print("✓ Export complete!\n")

# ============================================================
# RUN AND EXPORT - Combined workflow
# ============================================================

def run_and_export(func_names='all', num_runs=10, export_format='all'):
    """Run experiments and export results"""
    # Get function names
    if func_names == 'all':
        func_names = list(benchmark_functions.keys())
    elif isinstance(func_names, str):
        func_names = [func_names]
    
    print(f"\n🚀 Run & Export")
    print("="*60)
    print(f"Functions: {len(func_names)}")
    print(f"Runs: {num_runs}")
    print(f"Export: {export_format}")
    print("="*60 + "\n")
    
    # Run experiments
    results_dict = {}
    start = time.time()
    
    for i, func_name in enumerate(func_names, 1):
        print(f"[{i}/{len(func_names)}] {func_name}...", end=" ", flush=True)
        results, _ = run_multiple_experiments(func_name, num_runs)
        results_dict[func_name] = results
        print("✓")
    
    exec_time = time.time() - start
    print(f"\n✓ Completed in {exec_time:.2f}s")
    
    # Export
    export_results(results_dict, export_format)
    
    return results_dict

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n🛠️  GA Utilities - Simple & Fast")
        print("="*60)
        print("\nCommands:")
        print("\n1. Quick Test (single function)")
        print("   python utils.py test <function> [runs]")
        print("   Example: python utils.py test Sphere 10")
        print("\n2. Batch Test (multiple functions)")
        print("   python utils.py batch <func1> <func2> ... [runs]")
        print("   Example: python utils.py batch Sphere Rastrigin Ackley 10")
        print("\n3. Export Results")
        print("   python utils.py export <all|function> [runs] [format]")
        print("   Example: python utils.py export all 30 csv")
        print("   Example: python utils.py export Sphere 10 json")
        print("\nFormats: all, csv, json, md")
        print("\nAvailable functions:")
        for i, func in enumerate(benchmark_functions.keys(), 1):
            print(f"  {i:2d}. {func}")
        print("="*60 + "\n")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    # QUICK TEST
    if command == 'test':
        if len(sys.argv) < 3:
            print("❌ Usage: python utils.py test <function> [runs]")
            sys.exit(1)
        
        func_name = sys.argv[2]
        num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        quick_test(func_name, num_runs)
    
    # BATCH TEST
    elif command == 'batch':
        if len(sys.argv) < 3:
            print("❌ Usage: python utils.py batch <func1> <func2> ... [runs]")
            sys.exit(1)
        
        # Get function names (all args except last if it's a number)
        args = sys.argv[2:]
        if args[-1].isdigit():
            num_runs = int(args[-1])
            func_names = args[:-1]
        else:
            num_runs = 10
            func_names = args
        
        batch_test(func_names, num_runs)
    
    # EXPORT
    elif command == 'export':
        if len(sys.argv) < 3:
            print("❌ Usage: python utils.py export <all|function> [runs] [format]")
            sys.exit(1)
        
        selection = sys.argv[2]
        num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        export_format = sys.argv[4] if len(sys.argv) > 4 else 'all'
        
        run_and_export(selection, num_runs, export_format)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: test, batch, or export")
        sys.exit(1)
