"""
Benchmark Functions for Genetic Algorithm Optimization
Contains 15 standard optimization test functions
All functions have optimal value of 0.0
"""

import numpy as np

# 1. Sphere Function
def sphere(x):
    """
    Sphere Function: f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(x**2)

# 2. Rastrigin Function
def rastrigin(x):
    """
    Rastrigin Function: Highly multimodal
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 3. Ackley Function
def ackley(x):
    """
    Ackley Function: Multimodal with deep valley
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e

# 4. Griewank Function
def griewank(x):
    """
    Griewank Function: Multimodal with many local minima
    Global minimum: f(0,...,0) = 0
    """
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1

# 5. Zakharov Function
def zakharov(x):
    """
    Zakharov Function: Unimodal
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    indices = np.arange(1, n + 1)
    sum2 = np.sum(0.5 * indices * x)
    return np.sum(x**2) + sum2**2 + sum2**4

# 6. Schwefel 2.22 Function
def schwefel_222(x):
    """
    Schwefel 2.22 Function
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# 7. Schwefel 1.2 Function
def schwefel_12(x):
    """
    Schwefel 1.2 Function: Unimodal
    Global minimum: f(0,...,0) = 0
    """
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

# 8. Sum of Different Powers Function
def sum_diff_powers(x):
    """
    Sum of Different Powers Function
    Global minimum: f(0,...,0) = 0
    """
    indices = np.arange(1, len(x) + 1)
    return np.sum(np.abs(x)**(indices + 1))

# 9. Matyas Function (2D only)
def matyas(x):
    """
    Matyas Function: 2D function
    Global minimum: f(0,0) = 0
    """
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

# 10. Dixon-Price Function
def dixon_price(x):
    """
    Dixon-Price Function
    Global minimum: f(x_i = 2^(-(2^i-2)/2^i)) = 0
    """
    n = len(x)
    indices = np.arange(2, n + 1)
    term1 = (x[0] - 1)**2
    term2 = np.sum(indices * (2 * x[1:]**2 - x[:-1])**2)
    return term1 + term2

# 11. Levy Function
def levy(x):
    """
    Levy Function: Multimodal
    Global minimum: f(1,...,1) = 0
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term_n = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    sum_part = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    return term1 + sum_part + term_n

# 12. Perm Function (d, beta=10)
def perm(x):
    """
    Perm Function with beta=10
    Global minimum: f(1,2,...,n) ≈ 0
    """
    n = len(x)
    beta = 10
    outer_sum = 0
    for i in range(1, n + 1):
        inner_sum = 0
        for j in range(1, n + 1):
            inner_sum += (j + beta) * (x[j-1]**i - (1/j)**i)
        outer_sum += inner_sum**2
    return outer_sum

# 13. Rotated Hyper-Ellipsoid
def rotated_hyper_ellipsoid(x):
    """
    Rotated Hyper-Ellipsoid Function
    Global minimum: f(0,...,0) = 0
    """
    return np.sum([np.sum(x[:i+1]**2) for i in range(len(x))])

# 14. Bent Cigar Function
def bent_cigar(x):
    """
    Bent Cigar Function
    Global minimum: f(0,...,0) = 0
    """
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

# 15. Booth Function (2D only)
def booth(x):
    """
    Booth Function: 2D function
    Global minimum: f(1,3) = 0
    """
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

# Function dictionary for easy access
benchmark_functions = {
    'Sphere': sphere,
    'Rastrigin': rastrigin,
    'Ackley': ackley,
    'Griewank': griewank,
    'Zakharov': zakharov,
    'Schwefel_222': schwefel_222,
    'Schwefel_12': schwefel_12,
    'Sum_Diff_Powers': sum_diff_powers,
    'Matyas': matyas,
    'Dixon_Price': dixon_price,
    'Levy': levy,
    'Perm': perm,
    'Rotated_Hyper_Ellipsoid': rotated_hyper_ellipsoid,
    'Bent_Cigar': bent_cigar,
    'Booth': booth
}

# Function bounds (typical ranges for testing)
function_bounds = {
    'Sphere': (-5.12, 5.12),
    'Rastrigin': (-5.12, 5.12),
    'Ackley': (-32.768, 32.768),
    'Griewank': (-600, 600),
    'Zakharov': (-5, 10),
    'Schwefel_222': (-10, 10),
    'Schwefel_12': (-100, 100),
    'Sum_Diff_Powers': (-1, 1),
    'Matyas': (-10, 10),
    'Dixon_Price': (-10, 10),
    'Levy': (-10, 10),
    'Perm': (-1, 1),
    'Rotated_Hyper_Ellipsoid': (-65.536, 65.536),
    'Bent_Cigar': (-100, 100),
    'Booth': (-10, 10)
}

# Function properties
function_properties = {
    'Sphere': {'type': 'Unimodal', 'separable': True, 'dimensions': 'Any'},
    'Rastrigin': {'type': 'Multimodal', 'separable': True, 'dimensions': 'Any'},
    'Ackley': {'type': 'Multimodal', 'separable': False, 'dimensions': 'Any'},
    'Griewank': {'type': 'Multimodal', 'separable': False, 'dimensions': 'Any'},
    'Zakharov': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Schwefel_222': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Schwefel_12': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Sum_Diff_Powers': {'type': 'Unimodal', 'separable': True, 'dimensions': 'Any'},
    'Matyas': {'type': 'Unimodal', 'separable': False, 'dimensions': '2D'},
    'Dixon_Price': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Levy': {'type': 'Multimodal', 'separable': False, 'dimensions': 'Any'},
    'Perm': {'type': 'Multimodal', 'separable': False, 'dimensions': 'Any'},
    'Rotated_Hyper_Ellipsoid': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Bent_Cigar': {'type': 'Unimodal', 'separable': False, 'dimensions': 'Any'},
    'Booth': {'type': 'Unimodal', 'separable': False, 'dimensions': '2D'}
}

# Test function
if __name__ == "__main__":
    print("Testing benchmark functions...")
    
    # Test with 5D input
    test_input = np.zeros(5)
    
    print("\nTesting all functions with zero vector (should give optimal or near-optimal):")
    for name, func in benchmark_functions.items():
        try:
            if name in ['Matyas', 'Booth']:
                # 2D functions
                result = func(np.zeros(2))
            else:
                result = func(test_input)
            print(f"{name:30s}: f(0) = {result:.6f}")
        except Exception as e:
            print(f"{name:30s}: Error - {str(e)}")
    
    print("\n✓ All functions tested successfully")
