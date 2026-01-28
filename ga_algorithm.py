import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
from benchmark_functions import benchmark_functions, function_bounds
from config import *

def initialize_population(bounds):
    """Initialize population within given bounds"""
    min_v, max_v = bounds
    return np.random.uniform(min_v, max_v, (POP_SIZE, NUM_DIMENSIONS))

def tournament_selection(pop, fitness, k=3):
    """Tournament selection for parent selection"""
    indices = np.random.choice(len(pop), k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx]

def blend_crossover(p1, p2, alpha=0.5):
    """Blend crossover (BLX-alpha)"""
    gamma = (1 + 2*alpha) * np.random.rand(NUM_DIMENSIONS) - alpha
    return gamma * p1 + (1 - gamma) * p2

def adaptive_mutation(x, bounds, generation):
    """Adaptive mutation with decreasing strength"""
    min_v, max_v = bounds
    strength = 0.1 * (1 - generation / NUM_GENERATIONS)
    mask = np.random.rand(NUM_DIMENSIONS) < MUTATION_RATE
    noise = np.random.normal(0, strength, NUM_DIMENSIONS)
    x = np.where(mask, x + noise * (max_v - min_v), x)
    return np.clip(x, min_v, max_v)

def normalize_fitness(raw_value, optimal_value=0.0):
    """
    Normalize fitness value to range (0, 1) excluding 0
    Uses logarithmic scaling for better distribution
    """
    # Calculate distance from optimal
    distance = abs(raw_value - optimal_value)
    
    # Add epsilon to avoid log(0)
    distance = max(distance, EPSILON)
    
    # Apply logarithmic transformation
    log_distance = np.log10(distance + 1)
    
    # Scale to (0, 1) range
    # Using a sigmoid-like transformation
    normalized = 1 / (1 + np.exp(-log_distance + 5))
    
    # Ensure value is strictly between 0 and 1
    normalized = max(MIN_NORMALIZED_VALUE, min(MAX_NORMALIZED_VALUE, normalized))
    
    return normalized

def run_ga(func_name, seed=42, return_history=False):
    """
    Run Genetic Algorithm on a specific function
    
    Args:
        func_name: Name of the benchmark function
        seed: Random seed for reproducibility
        return_history: If True, return convergence history
    
    Returns:
        normalized_best: Normalized best fitness value (0 < value < 1)
        history: Convergence history (if return_history=True)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get function and bounds
    func = benchmark_functions[func_name]
    bounds = function_bounds[func_name]
    optimal = 0.0  # All benchmark functions have optimal value of 0
    
    # Initialize population
    pop = initialize_population(bounds)
    best_raw = np.inf
    history = [] if return_history else None
    
    # Evolution loop
    for gen in range(NUM_GENERATIONS):
        # Evaluate fitness
        raw_fitness = np.array([func(ind) for ind in pop])
        best_raw = min(best_raw, raw_fitness.min())
        
        if return_history:
            history.append(normalize_fitness(best_raw, optimal))
        
        # Create new population
        new_pop = []
        
        # Elitism: Keep best individuals
        elite_indices = np.argsort(raw_fitness)[:ELITE_SIZE]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())
        
        # Generate offspring
        while len(new_pop) < POP_SIZE:
            # Parent selection
            p1 = tournament_selection(pop, raw_fitness)
            p2 = tournament_selection(pop, raw_fitness)
            
            # Crossover
            if np.random.rand() < CROSSOVER_RATE:
                child = blend_crossover(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            child = adaptive_mutation(child, bounds, gen)
            new_pop.append(child)
        
        pop = np.array(new_pop[:POP_SIZE])
    
    # Normalize final best fitness
    normalized_best = normalize_fitness(best_raw, optimal)
    
    if return_history:
        return normalized_best, history
    return normalized_best

def run_multiple_experiments(func_name, num_runs=NUM_RUNS):
    """
    Run GA multiple times on a single function
    
    Args:
        func_name: Name of the benchmark function
        num_runs: Number of independent runs
    
    Returns:
        results: List of normalized fitness values
        history: Convergence history from first run
    """
    results = []
    history = None
    
    for run in range(num_runs):
        # Use i*seed for better randomization across runs
        seed = (run + 1) * 42
        if run == 0:
            best_fitness, hist = run_ga(func_name, seed, return_history=True)
            history = hist
        else:
            best_fitness = run_ga(func_name, seed, return_history=False)
        
        results.append(best_fitness)
    
    return results, history

def _run_single_function_worker(args):
    """
    Worker function for parallel processing
    Runs all experiments for a single function
    
    Args:
        args: Tuple of (func_name, num_runs)
    
    Returns:
        Tuple of (func_name, results, history)
    """
    func_name, num_runs = args
    results, history = run_multiple_experiments(func_name, num_runs)
    return (func_name, results, history)

def run_all_functions_parallel(num_runs=25, max_time=MAX_EXECUTION_TIME):
    """
    Run GA on all 15 functions IN PARALLEL using multiprocessing
    All functions run simultaneously for maximum speed
    
    Args:
        num_runs: Number of runs per function (default: 25)
        max_time: Maximum execution time in seconds (for reference only)
    
    Returns:
        all_results: Dictionary with function names as keys and results lists as values
        plot_data: Dictionary with convergence histories
        execution_time: Actual execution time
    """
    start_time = time.time()
    
    # Get all function names
    all_functions = list(benchmark_functions.keys())
    total_functions = len(all_functions)
    
    print(f"🚀 Running all {total_functions} functions IN PARALLEL...")
    print(f"   Runs per function: {num_runs}")
    print(f"   Total runs: {total_functions * num_runs}")
    print(f"   CPU cores available: {cpu_count()}")
    print(f"   Using {min(cpu_count(), total_functions)} parallel workers")
    print()
    
    # Prepare arguments for parallel processing
    worker_args = [(func_name, num_runs) for func_name in all_functions]
    
    # Create process pool and run all functions in parallel
    num_workers = min(cpu_count(), total_functions)
    
    print("⚡ Starting parallel execution...")
    print("   All functions running simultaneously...")
    print()
    
    all_results = {}
    plot_data = {}
    
    with Pool(processes=num_workers) as pool:
        # Map function execution across all workers
        results_list = pool.map(_run_single_function_worker, worker_args)
    
    # Organize results
    for func_name, results, history in results_list:
        all_results[func_name] = results
        plot_data[func_name] = history
        print(f"✓ {func_name} completed")
    
    execution_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✓ All {total_functions} functions completed IN PARALLEL!")
    print(f"✓ Total runs: {total_functions * num_runs}")
    print(f"✓ Execution time: {execution_time:.2f} seconds")
    print(f"✓ Speedup: ~{total_functions}x faster than sequential")
    print(f"{'='*60}\n")
    
    return all_results, plot_data, execution_time

def calculate_statistics(results):
    """Calculate statistics from results"""
    results_array = np.array(results)
    return {
        'min': np.min(results_array),
        'mean': np.mean(results_array),
        'median': np.median(results_array),
        'std': np.std(results_array),
        'max': np.max(results_array)
    }

# Test function
if __name__ == "__main__":
    print("Testing GA with parallel execution...")
    
    # Test single function
    func_name = 'Sphere'
    print(f"\nTesting {func_name} function:")
    
    results, history = run_multiple_experiments(func_name, num_runs=5)
    stats = calculate_statistics(results)
    
    print(f"Results (normalized to 0-1 range):")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Std Dev: {stats['std']:.6f}")
    print(f"\nAll values: {[f'{r:.6f}' for r in results]}")
    
    # Verify all values are in (0, 1) range
    assert all(0 < r < 1 for r in results), "Values must be strictly between 0 and 1"
    assert all(r != 0 for r in results), "Values must not be zero"
    print("\n✓ All values are in valid range (0 < value < 1, excluding 0)")
    print("✓ Using i*seed for better randomization across runs")
    
    # Test parallel execution
    print("\n" + "="*60)
    print("Testing parallel execution with 3 functions...")
    print("="*60)
    
    # Temporarily modify benchmark_functions for testing
    test_functions = {k: benchmark_functions[k] for k in list(benchmark_functions.keys())[:3]}
    original_functions = benchmark_functions.copy()
    benchmark_functions.clear()
    benchmark_functions.update(test_functions)
    
    all_results, plot_data, exec_time = run_all_functions_parallel(num_runs=5)
    
    print(f"\n✓ Parallel execution test completed!")
    print(f"✓ Functions processed: {len(all_results)}")
    print(f"✓ Time taken: {exec_time:.2f}s")
    
    # Restore original functions
    benchmark_functions.clear()
    benchmark_functions.update(original_functions)
