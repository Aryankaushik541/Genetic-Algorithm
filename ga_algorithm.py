import numpy as np
import random
import time
from tabulate import tabulate

# GA Parameters
NUM_DIMENSIONS = 5
POP_SIZE = 50
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_SIZE = 2

def initialize_population(bounds):
    min_v, max_v = bounds
    return np.random.uniform(min_v, max_v, (POP_SIZE, NUM_DIMENSIONS))

def tournament_selection(pop, fitness, k=3):
    indices = np.random.choice(len(pop), k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx]

def blend_crossover(p1, p2, alpha=0.5):
    gamma = (1 + 2*alpha) * np.random.rand(NUM_DIMENSIONS) - alpha
    return gamma * p1 + (1 - gamma) * p2

def adaptive_mutation(x, bounds, generation):
    min_v, max_v = bounds
    strength = 0.1 * (1 - generation / NUM_GENERATIONS)
    mask = np.random.rand(NUM_DIMENSIONS) < MUTATION_RATE
    noise = np.random.normal(0, strength, NUM_DIMENSIONS)
    x = np.where(mask, x + noise * (max_v - min_v), x)
    return np.clip(x, min_v, max_v)

def normalize_fitness(raw_values, optimal_value):
    """Better normalization that preserves relative differences"""
    adjusted = np.abs(np.array(raw_values) - optimal_value)
    epsilon = 1e-15
    adjusted = adjusted + epsilon
    log_adjusted = np.log10(adjusted + 1)
    
    if np.max(log_adjusted) == np.min(log_adjusted):
        return np.zeros_like(log_adjusted)
    
    normalized = (log_adjusted - np.min(log_adjusted)) / (np.max(log_adjusted) - np.min(log_adjusted))
    return normalized

def run_ga(func_info, seed):
    np.random.seed(seed)
    random.seed(seed)
    
    func = func_info["func"]
    bounds = func_info["bounds"]
    optimal = func_info["optimal"]
    
    pop = initialize_population(bounds)
    best_hist = []
    best_raw = np.inf
    
    for gen in range(NUM_GENERATIONS):
        raw_fitness = np.array([func(ind) for ind in pop])
        best_raw = min(best_raw, raw_fitness.min())
        best_hist.append(best_raw)
        
        new_pop = []
        elite_indices = np.argsort(raw_fitness)[:ELITE_SIZE]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())
        
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, raw_fitness)
            p2 = tournament_selection(pop, raw_fitness)
            
            if np.random.rand() < CROSSOVER_RATE:
                child = blend_crossover(p1, p2)
            else:
                child = p1.copy()
            
            child = adaptive_mutation(child, bounds, gen)
            new_pop.append(child)
        
        pop = np.array(new_pop[:POP_SIZE])
    
    return best_raw, best_hist, optimal

# Function info dictionary with optimal values
function_info = {
    'Sphere': {"func": sphere, "bounds": (-5.12, 5.12), "optimal": 0.0},
    'Rastrigin': {"func": rastrigin, "bounds": (-5.12, 5.12), "optimal": 0.0},
    'Ackley': {"func": ackley, "bounds": (-32.768, 32.768), "optimal": 0.0},
    'Griewank': {"func": griewank, "bounds": (-600, 600), "optimal": 0.0},
    'Zakharov': {"func": zakharov, "bounds": (-5, 10), "optimal": 0.0},
    'Schwefel_222': {"func": schwefel_222, "bounds": (-10, 10), "optimal": 0.0},
    'Schwefel_12': {"func": schwefel_12, "bounds": (-100, 100), "optimal": 0.0},
    'Sum_Diff_Powers': {"func": sum_diff_powers, "bounds": (-1, 1), "optimal": 0.0},
    'Dixon_Price': {"func": dixon_price, "bounds": (-10, 10), "optimal": 0.0},
    'Levy': {"func": levy, "bounds": (-10, 10), "optimal": 0.0},
    'Perm': {"func": perm, "bounds": (-1, 1), "optimal": 0.0},
    'Rotated_Hyper_Ellipsoid': {"func": rotated_hyper_ellipsoid, "bounds": (-65.536, 65.536), "optimal": 0.0},
    'Bent_Cigar': {"func": bent_cigar, "bounds": (-100, 100), "optimal": 0.0}
}

def run_multiple_experiments(num_runs=30):
    """Run GA on all benchmark functions multiple times"""
    all_results = []
    start_time = time.time()
    
    for func_name, func_info in function_info.items():
        print(f"\n🔄 Running {func_name}...")
        results = []
        
        for run in range(num_runs):
            seed = run + 42  # Different seed for each run
            best_fitness, _, _ = run_ga(func_info, seed)
            results.append(best_fitness)
            
            if (run + 1) % 10 == 0:
                print(f"   Completed {run + 1}/{num_runs} runs")
        
        # Calculate statistics
        results = np.array(results)
        stats = [
            func_name,
            np.min(results),
            np.mean(results),
            np.median(results),
            np.std(results)
        ]
        all_results.append(stats)
    
    execution_time = time.time() - start_time
    return all_results, execution_time

def display_individual_tables(results, execution_time):
    """Display individual table for each function"""
    print("\n" + "="*95)
    print(" GA PERFORMANCE STATISTICAL SUMMARY (30 RUNS) ".center(95))
    print("="*95)
    
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
    
    print(f"\n{'='*95}")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print("="*95)

def display_comparison_table(results):
    """Display comparison table for all functions"""
    print("\n" + "="*95)
    print(" COMPARISON TABLE ".center(95))
    print("="*95)
    print(tabulate(results,
                   headers=["Function", "Min", "Mean", "Median", "Std Dev"],
                   tablefmt="grid",
                   floatfmt=".6f"))

def get_best_function_info(results):
    """Find and display best performing function"""
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

# Main execution
if __name__ == "__main__":
    print("🚀 Starting GA Benchmark Tests...")
    
    # Run experiments
    results, exec_time = run_multiple_experiments(30)
    
    # Display results
    display_individual_tables(results, exec_time)
    display_comparison_table(results)
    get_best_function_info(results)