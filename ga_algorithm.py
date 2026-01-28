"""
Genetic Algorithm Implementation
"""

import numpy as np
import random
import time
from multiprocessing import Pool, cpu_count
from benchmark_functions import benchmark_functions, function_bounds
from config import *

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

def normalize_fitness(raw_value, optimal_value=0.0):
    distance = abs(raw_value - optimal_value)
    distance = max(distance, EPSILON)
    log_distance = np.log10(distance + 1)
    normalized = 1 / (1 + np.exp(-log_distance + 5))
    normalized = max(MIN_NORMALIZED_VALUE, min(MAX_NORMALIZED_VALUE, normalized))
    return normalized

def run_ga(func_name, seed=42, return_history=False):
    np.random.seed(seed)
    random.seed(seed)
    
    func = benchmark_functions[func_name]
    bounds = function_bounds[func_name]
    optimal = 0.0
    
    pop = initialize_population(bounds)
    best_raw = np.inf
    history = [] if return_history else None
    
    for gen in range(NUM_GENERATIONS):
        raw_fitness = np.array([func(ind) for ind in pop])
        best_raw = min(best_raw, raw_fitness.min())
        
        if return_history:
            history.append(normalize_fitness(best_raw, optimal))
        
        new_pop = []
        
        # Elitism
        elite_indices = np.argsort(raw_fitness)[:ELITE_SIZE]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())
        
        # Generate offspring
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
    
    normalized_best = normalize_fitness(best_raw, optimal)
    
    if return_history:
        return normalized_best, history
    return normalized_best

def run_multiple_experiments(func_name, num_runs=NUM_RUNS):
    results = []
    history = None
    
    for run in range(num_runs):
        seed = (run + 1) * 42
        if run == 0:
            best_fitness, hist = run_ga(func_name, seed, return_history=True)
            history = hist
        else:
            best_fitness = run_ga(func_name, seed, return_history=False)
        
        results.append(best_fitness)
    
    return results, history

def _run_single_function_worker(args):
    func_name, num_runs = args
    results, history = run_multiple_experiments(func_name, num_runs)
    return (func_name, results, history)

def run_all_functions_parallel(num_runs=25, max_time=MAX_EXECUTION_TIME):
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
    results_array = np.array(results)
    return {
        'min': np.min(results_array),
        'mean': np.mean(results_array),
        'median': np.median(results_array),
        'std': np.std(results_array),
        'max': np.max(results_array)
    }
