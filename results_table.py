from tabulate import tabulate
import numpy as np

def display_results_table(results, execution_time):
    print("\n" + "="*95)
    print(" GA PERFORMANCE STATISTICAL SUMMARY (30 RUNS) ".center(95))
    print("="*95)
    
    # Individual function tables
    for result in results:
        function_name = result[0]
        min_val = result[1]
        mean_val = result[2]
        median_val = result[3]
        std_dev = result[4]
        
        table_data = [
            ["Min", f"{min_val:.6f}"],
            ["Mean", f"{mean_val:.6f}"],
            ["Median", f"{median_val:.6f}"],
            ["Std Dev", f"{std_dev:.6f}"]
        ]
        
        print(f"\n📊 {function_name}")
        print(tabulate(table_data, 
                       headers=["Metric", "Value"],
                       tablefmt="grid"))
    
    # Summary table with all functions
    print(f"\n{'='*95}")
    print(" COMPARATIVE SUMMARY ".center(95))
    print("="*95)
    
    summary_data = []
    for result in results:
        summary_data.append([
            result[0],  # Function name
            f"{result[1]:.6f}",  # Min
            f"{result[2]:.6f}",  # Mean
            f"{result[3]:.6f}",  # Median
            f"{result[4]:.6f}"   # Std Dev
        ])
    
    print(tabulate(summary_data,
                   headers=["Function", "Min", "Mean", "Median", "Std Dev"],
                   tablefmt="grid"))
    
    print(f"\n{'='*95}")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print("="*95)

def get_best_function_info(results):
    best_function_idx = np.argmin([r[2] for r in results])
    best_function = results[best_function_idx][0]
    
    print(f"\n🏆 BEST PERFORMING FUNCTION: {best_function}")
    print(f"   Mean Performance: {results[best_function_idx][2]:.6f}")
    print(f"   Standard Deviation: {results[best_function_idx][4]:.6f}")
    
    print(f"\n📊 OVERALL STATISTICS:")
    all_means = [r[2] for r in results]
    print(f"   Average Mean Across All Functions: {np.mean(all_means):.6f}")
    print(f"   Best Mean Performance: {np.min(all_means):.6f}")
    print(f"   Worst Mean Performance: {np.max(all_means):.6f}")
    
    return best_function, best_function_idx

# Example usage
if __name__ == "__main__":
    # Sample data structure: [function_name, min, mean, median, std_dev]
    sample_results = [
        ["Sphere Function", 0.000001, 0.000045, 0.000032, 0.000023],
        ["Rastrigin Function", 2.345678, 5.678901, 4.567890, 1.234567],
        ["Ackley Function", 0.123456, 0.456789, 0.345678, 0.098765]
    ]
    
    display_results_table(sample_results, 45.67)
    get_best_function_info(sample_results)