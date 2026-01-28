"""
Optimized Configuration for Genetic Algorithm
"""

# GA Parameters - Optimized for better convergence
NUM_DIMENSIONS = 5
POP_SIZE = 100              # Increased for better diversity
NUM_GENERATIONS = 2000      # More generations for convergence
CROSSOVER_RATE = 0.9        # Higher crossover rate
MUTATION_RATE = 0.1         # Balanced mutation
ELITE_SIZE = 5              # Keep more elite solutions

# Experiment Settings
NUM_RUNS = 30               # For individual function
COMBINED_RUNS = 25          # For all functions together
MAX_EXECUTION_TIME = 30     # Seconds

# Display
TABLE_FORMAT = 'grid'
DECIMAL_PLACES = 6
FIGURE_SIZE = (16, 12)
DPI = 100

# Normalization (not used in optimized version)
EPSILON = 1e-10
MIN_NORMALIZED_VALUE = 1e-8
MAX_NORMALIZED_VALUE = 0.999999
