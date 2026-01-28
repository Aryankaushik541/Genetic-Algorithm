import time
import numpy as np
from benchmark_functions import functions_list
from ga_algorithm import run_ga

def run_experiment(num_runs=30):
    results = []
    plot_data = {}
    all_raw_scores = {}
    
    start = time.time()
    print(f"Starting Experiment: 15 Functions, {num_runs} Runs Each\n")
    
    for f_info in functions_list:
        name = f_info["name"]
        raw_scores = []
        history = None
        
        for r in range(1, num_runs + 1):
            print(f"\rOptimizing {name:12} | Run {r:02}/{num_runs}", end="", flush=True)
            best_raw, hist, optimal = run_ga(f_info, seed=r)
            raw_scores.append(best_raw)
            if r == 1:
                history = hist
        
        all_raw_scores[name] = raw_scores
        
        print(f"\rOptimized {name:12} | Done.{' '*20}")
        
        # Store RAW values instead of normalized
        results.append([
            name, 
            np.min(raw_scores), 
            np.mean(raw_scores),
            np.median(raw_scores), 
            np.std(raw_scores)
        ])
        
        plot_data[name] = history
    
    end = time.time()
    execution_time = end - start
    
    return results, plot_data, all_raw_scores, execution_time

if __name__ == "__main__":
    results, plot_data, all_scores, exec_time = run_experiment()
    
    # Find best function (lowest mean)
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
    print(f"\nTotal Execution Time: {exec_time:.2f} seconds")