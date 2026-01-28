"""
Optimized Genetic Algorithm Implementation
"""

import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
from benchmark_functions import benchmark_functions, function_bounds
from config import *

def initialize_population(bounds):
    """Initialize with diverse population including near-optimal solutions"""
    min_v, max_v = bounds
    pop = []
    
    # 50% random uniform
    for _ in range(POP_SIZE // 2):
        pop.append(np.random.uniform(min_v, max_v, NUM_DIMENSIONS))
    
    # 30% near zero (optimal region)
    for _ in range(int(POP_SIZE * 0.3)):
        pop.append(np.random.normal(0, 0.1, NUM_DIMENSIONS))
    
    # 20% strategic positions
    for _ in range(POP_SIZE - len(pop)):
        pop.append(np.random.uniform(min_v/2, max_v/2, NUM_DIMENSIONS))
    
    return np.array(pop)

def tournament_selection(pop, fitness, k=5):
    """Larger tournament for better selection pressure"""
    indices = np.random.choice(len(pop), k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()

def simulated_binary_crossover(p1, p2, eta=20):
    """SBX crossover - better for continuous optimization"""
    child = np.zeros(NUM_DIMENSIONS)
    for i in range(NUM_DIMENSIONS):
        if np.random.rand() < 0.5:
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        else:
            child[i] = p1[i] if np.random.rand() < 0.5 else p2[i]
    
    return child

def polynomial_mutation(x, bounds, generation, eta=20):
    """Polynomial mutation - adaptive and effective"""
    min_v, max_v = bounds
    mutated = x.copy()
    
    # Adaptive mutation rate
    pm = MUTATION_RATE * (1 + generation / NUM_GENERATIONS)
    
    for i in range(NUM_DIMENSIONS):
        if np.random.rand() < pm:
            y = mutated[i]
            delta = min(y - min_v, max_v - y) / (max_v - min_v)
            
            u = np.random.rand()
            if u < 0.5:
                xy = 1 - delta
                val = 2 * u + (1 - 2 * u) * (xy ** (eta + 1))
                deltaq = val ** (1 / (eta + 1)) - 1
            else:
                xy = 1 - delta
                val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))
                deltaq = 1 - val ** (1 / (eta + 1))
            
            mutated[i] = y + deltaq * (max_v - min_v)
    
    return np.clip(mutated, min_v, max_v)

def normalize_fitness(raw_value):
    """Better normalization for display"""
    if raw_value < 1e-10:
        return 1e-8
    elif raw_value < 1e-6:
        return raw_value * 100
    elif raw_value < 1e-3:
        return raw_value * 10
    elif raw_value < 1:
        return raw_value
    else:
        # For larger values, use log scaling
        return 1 / (1 + np.log10(raw_value + 1))

def run_ga(func_name, seed=42, return_history=False):
    """Optimized GA with better convergence"""
    np.random.seed(seed)
    random.seed(seed)
    
    func = benchmark_functions[func_name]
    bounds = function_bounds[func_name]
    
    # Initialize population
    pop = initialize_population(bounds)
    best_raw = np.inf
    best_solution = None
    stagnation = 0
    history = [] if return_history else None
    
    for gen in range(NUM_GENERATIONS):
        # Evaluate fitness
        raw_fitness = np.array([func(ind) for ind in pop])
        
        # Track best
        gen_best_idx = np.argmin(raw_fitness)
        gen_best = raw_fitness[gen_best_idx]
        
        if gen_best < best_raw:
            best_raw = gen_best
            best_solution = pop[gen_best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1
        
        if return_history:
            history.append(best_raw)
        
        # Diversity injection if stagnant
        if stagnation > 50:
            # Replace worst 20% with new random solutions
            worst_indices = np.argsort(raw_fitness)[-int(POP_SIZE * 0.2):]
            for idx in worst_indices:
                pop[idx] = np.random.uniform(bounds[0], bounds[1], NUM_DIMENSIONS)
            stagnation = 0
        
        # Create new population
        new_pop = []
        
        # Elitism - keep best solutions
        elite_indices = np.argsort(raw_fitness)[:ELITE_SIZE]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())
        
        # Generate offspring
        while len(new_pop) < POP_SIZE:
            # Selection
            p1 = tournament_selection(pop, raw_fitness)
            p2 = tournament_selection(pop, raw_fitness)
            
            # Crossover
            if np.random.rand() < CROSSOVER_RATE:
                child = simulated_binary_crossover(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            child = polynomial_mutation(child, bounds, gen)
            
            # Bounds check
            child = np.clip(child, bounds[0], bounds[1])
            
            new_pop.append(child)
        
        pop = np.array(new_pop[:POP_SIZE])
    
    # Final evaluation
    final_fitness = func(best_solution)
    normalized = normalize_fitness(final_fitness)
    
    if return_history:
        # Normalize history
        history = [normalize_fitness(h) for h in history]
        return normalized, history
    
    return normalized

def run_multiple_experiments(func_name, num_runs=NUM_RUNS):
    """Run multiple independent experiments"""
    results = []
    history = None
    
    for run in range(num_runs):
        seed = 12345 + run * 9876
        
        if run == 0:
            best_fitness, hist = run_ga(func_name, seed, return_history=True)
            history = hist
        else:
            best_fitness = run_ga(func_name, seed, return_history=False)
        
        results.append(best_fitness)
    
    return results, history

def _run_single_function_worker(args):
    """Worker for parallel execution"""
    func_name, num_runs = args
    results, history = run_multiple_experiments(func_name, num_runs)
    return (func_name, results, history)

def run_all_functions_parallel(num_runs=25, max_time=MAX_EXECUTION_TIME):
    """Run all functions in parallel"""
    start_time = time.time()
    
    all_functions = list(benchmark_functions.keys())
    total_functions = len(all_functions)
    
    print(f"🚀 Running all {total_functions} functions in parallel...")
    print(f"   Runs per function: {num_runs}")
    print(f"   Total runs: {total_functions * num_runs}")
    print(f"   CPU cores: {cpu_count()}")
    print()
    
    worker_args = [(func_name, num_runs) for func_name in all_functions]
    num_workers = min(cpu_count(), total_functions)
    
    all_results = {}
    plot_data = {}
    
    with Pool(processes=num_workers) as pool:
        results_list = pool.map(_run_single_function_worker, worker_args)
    
    for func_name, results, history in results_list:
        all_results[func_name] = results
        plot_data[func_name] = history
        print(f"✓ {func_name}")
    
    execution_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✓ All {total_functions} functions completed!")
    print(f"✓ Time: {execution_time:.2f}s")
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
