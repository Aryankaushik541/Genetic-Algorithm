import numpy as np

# ==============================================================
#   15 BENCHMARK OPTIMIZATION TEST FUNCTIONS
#   Ordered as per your list
# ==============================================================


# 1. Sphere Function
# ----------------------------------------------------------
def sphere(x):
   
    return np.sum(x ** 2)


# 2. Sum of Squares (Weighted Sphere)
# ----------------------------------------------------------
def sum_of_squares(x):
    """
    Sum of Squares Function
    Formula:   f(x) = sum(i * x_i^2),  i = 1..n
    Type:      Unimodal | Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-10, 10]
    """
    indices = np.arange(1, len(x) + 1)
    return np.sum(indices * x ** 2)


# 3. Schwefel 2.22
# ----------------------------------------------------------
def schwefel_222(x):
    """
    Schwefel 2.22 Function
    Formula:   f(x) = sum(|x_i|) + prod(|x_i|)
    Type:      Unimodal | Non-Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-10, 10]
    """
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


# 4. Step Function
# ----------------------------------------------------------
def step(x):
    """
    Step Function
    Formula:   f(x) = sum(floor(|x_i|))
    Type:      Discontinuous | Separable
    Minimum:   f(x) = 0  for all x in [-1, 1]^n
    Bounds:    [-100, 100]
    """
    return np.sum(np.floor(np.abs(x)))


# 5. Rosenbrock Function
# ----------------------------------------------------------
def rosenbrock(x):
    """
    Rosenbrock Function (Banana Function)
    Formula:   f(x) = sum[ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
    Type:      Unimodal | Non-Separable
    Minimum:   f(1, ..., 1) = 0
    Bounds:    [-5, 10]
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# 6. Rastrigin Function
# ----------------------------------------------------------
def rastrigin(x):
    """
    Rastrigin Function
    Formula:   f(x) = 10n + sum[ x_i^2 - 10*cos(2*pi*x_i) ]
    Type:      Multimodal | Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-5.12, 5.12]
    """
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# 7. Ackley Function
# ----------------------------------------------------------
def ackley(x):
    """
    Ackley Function
    Formula:   f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n))
                       - exp(sum(cos(2*pi*x_i))/n) + 20 + e
    Type:      Multimodal | Non-Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-32.768, 32.768]
    """
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e


# 8. Griewank Function
# ----------------------------------------------------------
def griewank(x):
    """
    Griewank Function
    Formula:   f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i / sqrt(i)))
    Type:      Multimodal | Non-Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-600, 600]
    """
    sum_part = np.sum(x ** 2) / 4000
    indices = np.arange(1, len(x) + 1)
    prod_part = np.prod(np.cos(x / np.sqrt(indices)))
    return 1 + sum_part - prod_part


# 9. Lévy Function
# ----------------------------------------------------------
def levy(x):
    """
    Lévy Function
    Formula:   w_i = 1 + (x_i - 1)/4
               f(x) = sin^2(pi*w_1)
                     + sum[ (w_i-1)^2 * (1 + 10*sin^2(pi*w_i + 1)) ]
                     + (w_n-1)^2 * (1 + sin^2(2*pi*w_n))
    Type:      Multimodal | Non-Separable
    Minimum:   f(1, ..., 1) = 0
    Bounds:    [-10, 10]
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term_mid = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term_n = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term_mid + term_n


# 10. Zakharov Function
# ----------------------------------------------------------
def zakharov(x):
    """
    Zakharov Function
    Formula:   S = sum(0.5 * i * x_i)
               f(x) = sum(x_i^2) + S^2 + S^4
    Type:      Unimodal | Non-Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-5, 10]
    """
    indices = np.arange(1, len(x) + 1)
    S = np.sum(0.5 * indices * x)
    return np.sum(x ** 2) + S ** 2 + S ** 4


# 11. Dixon–Price Function
# ----------------------------------------------------------
def dixon_price(x):
    """
    Dixon–Price Function
    Formula:   f(x) = (x_1 - 1)^2 + sum[ i*(2*x_i^2 - x_{i-1})^2 ], i=2..n
    Type:      Unimodal | Non-Separable
    Minimum:   f(x*) = 0,  x_i* = 2^(-(2^i - 2) / 2^i)
    Bounds:    [-10, 10]
    """
    term1 = (x[0] - 1) ** 2
    indices = np.arange(2, len(x) + 1)
    term2 = np.sum(indices * (2 * x[1:] ** 2 - x[:-1]) ** 2)
    return term1 + term2


# 12. Bent Cigar Function
# ----------------------------------------------------------
def bent_cigar(x):
    """
    Bent Cigar Function
    Formula:   f(x) = x_1^2 + 10^6 * sum(x_i^2),  i=2..n
    Type:      Unimodal | Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-100, 100]
    """
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


# 13. High-Conditioned Elliptic Function
# ----------------------------------------------------------
def elliptic(x):
    """
    High-Conditioned Elliptic Function
    Formula:   f(x) = sum[ 10^(6*(i-1)/(n-1)) * x_i^2 ],  i=1..n
    Type:      Unimodal | Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-100, 100]
    Note:      Condition number = 10^6. Each dimension is scaled
               exponentially — a strong test for ill-conditioning.
    """
    n = len(x)
    indices = np.arange(n)                          # 0, 1, ..., n-1
    exponents = 1e6 ** (indices / max(n - 1, 1))    # 10^(6*(i)/(n-1))
    return np.sum(exponents * x ** 2)


# 14. Alpine Function (Alpine No.1)
# ----------------------------------------------------------
def alpine(x):
    """
    Alpine No.1 Function
    Formula:   f(x) = sum( |x_i * sin(x_i) + 0.1 * x_i| )
    Type:      Multimodal | Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-10, 10]
    """
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


# 15. Salomon Function
# ----------------------------------------------------------
def salomon(x):
    """
    Salomon Function
    Formula:   r = sqrt(sum(x_i^2))
               f(x) = 1 - cos(2*pi*r) + 0.1*r
    Type:      Multimodal | Non-Separable
    Minimum:   f(0, ..., 0) = 0
    Bounds:    [-10, 10]
    Note:      Concentric rings of local minima around the origin.
    """
    r = np.sqrt(np.sum(x ** 2))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r


# ==============================================================
#   LOOKUP DICTIONARIES
# ==============================================================

benchmark_functions = {
    "Sphere":              sphere,
    "Sum_of_Squares":      sum_of_squares,
    "Schwefel_222":        schwefel_222,
    "Step":                step,
    "Rosenbrock":          rosenbrock,
    "Rastrigin":           rastrigin,
    "Ackley":              ackley,
    "Griewank":            griewank,
    "Levy":                levy,
    "Zakharov":            zakharov,
    "Dixon_Price":         dixon_price,
    "Bent_Cigar":          bent_cigar,
    "Elliptic":            elliptic,
    "Alpine":              alpine,
    "Salomon":             salomon,
}

function_bounds = {
    "Sphere":              (-5.12,   5.12),
    "Sum_of_Squares":      (-10,     10),
    "Schwefel_222":        (-10,     10),
    "Step":                (-100,    100),
    "Rosenbrock":          (-5,      10),
    "Rastrigin":           (-5.12,   5.12),
    "Ackley":              (-32.768, 32.768),
    "Griewank":            (-600,    600),
    "Levy":                (-10,     10),
    "Zakharov":            (-5,      10),
    "Dixon_Price":         (-10,     10),
    "Bent_Cigar":          (-100,    100),
    "Elliptic":            (-100,    100),
    "Alpine":              (-10,     10),
    "Salomon":             (-10,     10),
}

function_properties = {
    "Sphere":              {"type": "Unimodal",       "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Sum_of_Squares":      {"type": "Unimodal",       "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Schwefel_222":        {"type": "Unimodal",       "separable": False, "min_at": "(0,...,0)",      "min_val": 0},
    "Step":                {"type": "Discontinuous",  "separable": True,  "min_at": "[-1,1]^n",      "min_val": 0},
    "Rosenbrock":          {"type": "Unimodal",       "separable": False, "min_at": "(1,...,1)",      "min_val": 0},
    "Rastrigin":           {"type": "Multimodal",     "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Ackley":              {"type": "Multimodal",     "separable": False, "min_at": "(0,...,0)",      "min_val": 0},
    "Griewank":            {"type": "Multimodal",     "separable": False, "min_at": "(0,...,0)",      "min_val": 0},
    "Levy":                {"type": "Multimodal",     "separable": False, "min_at": "(1,...,1)",      "min_val": 0},
    "Zakharov":            {"type": "Unimodal",       "separable": False, "min_at": "(0,...,0)",      "min_val": 0},
    "Dixon_Price":         {"type": "Unimodal",       "separable": False, "min_at": "x_i=2^(-(2^i-2)/2^i)", "min_val": 0},
    "Bent_Cigar":          {"type": "Unimodal",       "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Elliptic":            {"type": "Unimodal",       "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Alpine":              {"type": "Multimodal",     "separable": True,  "min_at": "(0,...,0)",      "min_val": 0},
    "Salomon":             {"type": "Multimodal",     "separable": False, "min_at": "(0,...,0)",      "min_val": 0},
}


# ==============================================================
#   TEST
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  BENCHMARK FUNCTIONS — VERIFICATION TEST (5D)")
    print("=" * 60)

    test_zero = np.zeros(5)
    test_one  = np.ones(5)

    print(f"\n{'#':<4} {'Function':<22} {'f(0...0)':<14} {'f(1...1)':<14} {'Status'}")
    print("-" * 60)

    for i, (name, func) in enumerate(benchmark_functions.items(), 1):
        try:
            val_zero = func(test_zero)
            val_one  = func(test_one)

            # Rosenbrock & Levy have minimum at (1,...,1)
            if name in ("Rosenbrock", "Levy"):
                status = "✓" if np.isclose(val_one, 0.0, atol=1e-10) else "✗"
            else:
                status = "✓" if np.isclose(val_zero, 0.0, atol=1e-10) else "✗"

            print(f"{i:<4} {name:<22} {val_zero:<14.6f} {val_one:<14.6f} {status}")
        except Exception as e:
            print(f"{i:<4} {name:<22} ERROR: {e}")

    print("-" * 60)
    print("  ✓ All 15 benchmark functions loaded & tested.\n")