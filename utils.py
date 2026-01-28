"""
Simple GA Utilities
Quick test, batch test, and export results
"""

import json
import csv
import sys
import time
from datetime import datetime
from ga_algorithm import run_multiple_experiments, calculate_statistics
from benchmark_functions import benchmark_functions

def quick_test(func_name, num_runs=10):
    """Test single function"""
    print(f"\n🔬 {func_name}")
    print("="*60)
    
    start = time.time()
    results, _ = run_multiple_experiments(func_name, num_runs)
    exec_time = time.time() - start
    stats = calculate_statistics(results)
    
    print(f"Runs: {num_runs} | Time: {exec_time:.2f}s\n")
    print(f"Min:    {stats['min']:.6f}")
    print(f"Mean:   {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Std:    {stats['std']:.6f}")
    print(f"Max:    {stats['max']:.6f}")
    print("="*60)

def batch_test(func_names, num_runs=10):
    """Test multiple functions"""
    print(f"\n📊 Batch Test: {len(func_names)} functions")
    print("="*60)
    
    all_results = {}
    start = time.time()
    
    for i, func_name in enumerate(func_names, 1):
        print(f"[{i}/{len(func_names)}] {func_name}...", end=" ")
        results, _ = run_multiple_experiments(func_name, num_runs)
        all_results[func_name] = calculate_statistics(results)
        print("✓")
    
    exec_time = time.time() - start
    
    print(f"\n✓ All completed in {exec_time:.2f}s\n")
    print("Comparison:")
    print("-"*60)
    
    # Sort by mean
    sorted_funcs = sorted(all_results.items(), key=lambda x: x[1]['mean'])
    
    for i, (name, stats) in enumerate(sorted_funcs, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{emoji} {name:25s} Mean: {stats['mean']:.6f}")
    
    print("="*60)

def export_results(func_names, num_runs=30, formats='all'):
    """Run and export results"""
    print(f"\n🚀 Run & Export")
    print("="*60)
    print(f"Functions: {len(func_names)}")
    print(f"Runs: {num_runs}")
    print(f"Export: {formats}")
    print("="*60 + "\n")
    
    all_results = {}
    
    for i, func_name in enumerate(func_names, 1):
        print(f"[{i}/{len(func_names)}] {func_name}...", end=" ")
        results, _ = run_multiple_experiments(func_name, num_runs)
        all_results[func_name] = {
            'results': results,
            'stats': calculate_statistics(results)
        }
        print("✓")
    
    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📤 Exporting...")
    
    if formats in ['csv', 'all']:
        filename = f"ga_results_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Function', 'Min', 'Mean', 'Median', 'Std', 'Max'])
            for func_name, data in all_results.items():
                stats = data['stats']
                writer.writerow([
                    func_name,
                    f"{stats['min']:.6f}",
                    f"{stats['mean']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['max']:.6f}"
                ])
        print(f"✓ CSV: {filename}")
    
    if formats in ['json', 'all']:
        filename = f"ga_results_{timestamp}.json"
        export_data = {}
        for func_name, data in all_results.items():
            export_data[func_name] = {
                'results': [float(r) for r in data['results']],
                'statistics': {k: float(v) for k, v in data['stats'].items()}
            }
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ JSON: {filename}")
    
    if formats in ['md', 'all']:
        filename = f"ga_results_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write("# GA Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Runs per function:** {num_runs}\n\n")
            f.write("## Results\n\n")
            f.write("| Function | Min | Mean | Median | Std | Max |\n")
            f.write("|----------|-----|------|--------|-----|-----|\n")
            for func_name, data in all_results.items():
                stats = data['stats']
                f.write(f"| {func_name} | {stats['min']:.6f} | {stats['mean']:.6f} | "
                       f"{stats['median']:.6f} | {stats['std']:.6f} | {stats['max']:.6f} |\n")
        print(f"✓ Markdown: {filename}")
    
    print("✓ Export complete!")

def show_help():
    print("""
GA Utilities - Simple Testing & Export

USAGE:
  python utils.py <command> [args]

COMMANDS:
  test <function> <runs>           Quick test single function
  batch <func1> <func2> ... <runs> Test multiple functions
  export <all|func> <runs> [fmt]   Run & export results

EXAMPLES:
  python utils.py test Sphere 10
  python utils.py batch Sphere Rastrigin Ackley 10
  python utils.py export all 30
  python utils.py export all 20 csv
  python utils.py export Sphere 10 json

FORMATS:
  csv  - CSV file
  json - JSON file
  md   - Markdown file
  all  - All formats (default)
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "test":
        if len(sys.argv) < 4:
            print("Usage: python utils.py test <function> <runs>")
            sys.exit(1)
        func_name = sys.argv[2]
        num_runs = int(sys.argv[3])
        quick_test(func_name, num_runs)
    
    elif cmd == "batch":
        if len(sys.argv) < 4:
            print("Usage: python utils.py batch <func1> <func2> ... <runs>")
            sys.exit(1)
        func_names = sys.argv[2:-1]
        num_runs = int(sys.argv[-1])
        batch_test(func_names, num_runs)
    
    elif cmd == "export":
        if len(sys.argv) < 4:
            print("Usage: python utils.py export <all|function> <runs> [format]")
            sys.exit(1)
        
        target = sys.argv[2]
        num_runs = int(sys.argv[3])
        fmt = sys.argv[4] if len(sys.argv) > 4 else 'all'
        
        if target.lower() == 'all':
            func_names = list(benchmark_functions.keys())
        else:
            func_names = [target]
        
        export_results(func_names, num_runs, fmt)
    
    else:
        print(f"Unknown command: {cmd}")
        show_help()
        sys.exit(1)
