"""
Genetic Algorithm Configuration Parameters
"""

# GA Parameters
NUM_DIMENSIONS = 5
POP_SIZE = 50
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_SIZE = 2

# Execution Parameters
MAX_EXECUTION_TIME = 20  # Maximum execution time in seconds
NUM_RUNS = 30  # Number of runs for statistical analysis

# Normalization Parameters
EPSILON = 1e-10  # Small value to avoid division by zero
MIN_NORMALIZED_VALUE = 0.001  # Minimum normalized value (> 0)
MAX_NORMALIZED_VALUE = 0.999  # Maximum normalized value (< 1)

# Visualization Parameters
FIGURE_SIZE = (16, 12)
DPI = 100
PLOT_TOP_N_FUNCTIONS = 5  # Number of top functions to show in convergence plot
