"""
Results Export Utility
Export GA results to various formats (CSV, JSON, Markdown)
"""

import json
import csv
import time
from datetime import datetime
from ga_algorithm import run_multiple_experiments, calculate_statistics
from benchmark_functions import benchmark_functions

def export_to_csv(results_dict, filename='ga_results.csv'):
    """
    Export results to CSV file
    
    Args:
        results_dict: Dictionary with function names as keys and results as values
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Function', 'Min', 'Mean', 'Median', 'Std Dev', 'Max', 'Runs'])
        
        # Data rows
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
    
    print(f"✓ Results exported to {filename}")

def export_to_json(results_dict, filename='ga_results.json'):
    """
    Export results to JSON file
    
    Args:
        results_dict: Dictionary with function names as keys and results as values
        filename: Output JSON filename
    """
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'total_functions': len(results_dict),
        'results': {}
    }
    
    for func_name, results in results_dict.items():
        stats = calculate_statistics(results)
        export_data['results'][func_name] = {
            'statistics': {
                'min': float(stats['min']),
                'mean': float(stats['mean']),
                'median': float(stats['median']),
                'std': float(stats['std']),
                'max': float(stats['max'])
            },
            'raw_values': [float(r) for r in results],
            'num_runs': len(results)
        }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Results exported to {filename}")

def export_to_markdown(results_dict, filename='ga_results.md'):
    """
    Export results to Markdown file
    
    Args:
        results_dict: Dictionary with function names as keys and results as values
        filename: Output Markdown filename
    """
    with open(filename, 'w') as f:
        # Header
        f.write("# Genetic Algorithm Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Functions:** {len(results_dict)}\n\n")
        
        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Rank | Function | Min | Mean | Median | Std Dev |\n")
        f.write("|------|----------|-----|------|--------|----------|\n")
        
        # Sort by mean performance
        sorted_results = sorted(results_dict.items(), 
                               key=lambda x: calculate_statistics(x[1])['mean'])
        
        for i, (func_name, results) in enumerate(sorted_results, 1):
            stats = calculate_statistics(results)
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            f.write(f"| {rank_emoji} | {func_name} | {stats['min']:.6f} | "
                   f"{stats['mean']:.6f} | {stats['median']:.6f} | {stats['std']:.6f} |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        
        for func_name, results in sorted_results:
            stats = calculate_statistics(results)
            f.write(f"### {func_name}\n\n")
            f.write(f"- **Best (Min):** {stats['min']:.6f}\n")
            f.write(f"- **Average (Mean):** {stats['mean']:.6f}\n")
            f.write(f"- **Median:** {stats['median']:.6f}\n")
            f.write(f"- **Std Deviation:** {stats['std']:.6f}\n")
            f.write(f"- **Worst (Max):** {stats['max']:.6f}\n")
            f.write(f"- **Number of Runs:** {len(results)}\n\n")
        
        # Best performers
        f.write("## Top 5 Performers\n\n")
        for i, (func_name, results) in enumerate(sorted_results[:5], 1):
            stats = calculate_statistics(results)
            f.write(f"{i}. **{func_name}** - Mean: {stats['mean']:.6f}, "
                   f"Std: {stats['std']:.6f}\n")
    
    print(f"✓ Results exported to {filename}")

def export_all_formats(results_dict, base_filename='ga_results'):
    """
    Export results to all formats (CSV, JSON, Markdown)
    
    Args:
        results_dict: Dictionary with function names as keys and results as values
        base_filename: Base filename (extensions will be added)
    """
    print("\n" + "="*60)
    print(" EXPORTING RESULTS ".center(60))
    print("="*60 + "\n")
    
    export_to_csv(results_dict, f"{base_filename}.csv")
    export_to_json(results_dict, f"{base_filename}.json")
    export_to_markdown(results_dict, f"{base_filename}.md")
    
    print("\n✓ All exports complete!")
    print(f"  - {base_filename}.csv")
    print(f"  - {base_filename}.json")
    print(f"  - {base_filename}.md")
    print("="*60 + "\n")

def run_and_export(function_names=None, num_runs=30, export_format='all'):
    """
    Run experiments and export results
    
    Args:
        function_names: List of function names (None = all)
        num_runs: Number of runs per function
        export_format: 'csv', 'json', 'md', or 'all'
    """
    if function_names is None:
        function_names = list(benchmark_functions.keys())
    
    print("\n" + "="*60)
    print(" RUNNING EXPERIMENTS ".center(60))
    print("="*60 + "\n")
    print(f"Functions: {len(function_names)}")
    print(f"Runs per function: {num_runs}")
    print()
    
    results_dict = {}
    start_time = time.time()
    
    for func_name in function_names:
        print(f"Processing {func_name}...", end=" ", flush=True)
        results, _ = run_multiple_experiments(func_name, num_runs)
        results_dict[func_name] = results
        print("✓")
    
    exec_time = time.time() - start_time
    print(f"\n✓ All experiments completed in {exec_time:.2f} seconds")
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"ga_results_{timestamp}"
    
    if export_format == 'all':
        export_all_formats(results_dict, base_filename)
    elif export_format == 'csv':
        export_to_csv(results_dict, f"{base_filename}.csv")
    elif export_format == 'json':
        export_to_json(results_dict, f"{base_filename}.json")
    elif export_format == 'md':
        export_to_markdown(results_dict, f"{base_filename}.md")
    
    return results_dict

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line usage
        format_arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        print("\nExport Results Utility")
        print("="*60)
        print(f"Format: {format_arg}")
        print(f"Runs: {num_runs}")
        
        # Run quick test with 3 functions
        test_functions = ['Sphere', 'Rastrigin', 'Ackley']
        run_and_export(test_functions, num_runs, format_arg)
    else:
        print("\nExport Results Utility")
        print("="*60)
        print("\nUsage:")
        print("  python export_results.py [format] [num_runs]")
        print("\nFormats:")
        print("  all  - Export to CSV, JSON, and Markdown (default)")
        print("  csv  - Export to CSV only")
        print("  json - Export to JSON only")
        print("  md   - Export to Markdown only")
        print("\nExamples:")
        print("  python export_results.py all 30")
        print("  python export_results.py csv 20")
        print("  python export_results.py json 10")
        print("\n" + "="*60 + "\n")
