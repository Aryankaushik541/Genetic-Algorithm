"""
Quick Test - Verify GA Optimization
"""

from ga_algorithm import run_ga
from benchmark_functions import benchmark_functions

print("\n" + "="*60)
print(" QUICK OPTIMIZATION TEST ".center(60))
print("="*60 + "\n")

# Test top 5 functions
test_functions = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Zakharov']

print("Testing with optimized GA...\n")

for func_name in test_functions:
    # Run single test
    result = run_ga(func_name, seed=12345, return_history=False)
    
    # Get raw value for comparison
    from benchmark_functions import benchmark_functions, function_bounds
    import numpy as np
    
    # Quick raw test
    np.random.seed(12345)
    func = benchmark_functions[func_name]
    bounds = function_bounds[func_name]
    test_point = np.zeros(5)  # Optimal point
    raw_optimal = func(test_point)
    
    print(f"{func_name:20s} | Normalized: {result:.8f} | Raw@Zero: {raw_optimal:.2e}")

print("\n" + "="*60)
print("Lower values = Better performance")
print("Target: Values should be very small (< 0.001 for easy functions)")
print("="*60 + "\n")
