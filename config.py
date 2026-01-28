"""
Simple Configuration for Genetic Algorithm
"""

# GA Parameters
NUM_DIMENSIONS = 5
POP_SIZE = 50
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_SIZE = 2

# Experiment Settings
NUM_RUNS = 30              # For individual function
COMBINED_RUNS = 25         # For all functions together
MAX_EXECUTION_TIME = 20    # Seconds

# Display
TABLE_FORMAT = 'grid'
DECIMAL_PLACES = 6
FIGURE_SIZE = (16, 12)
DPI = 100

# Normalization
EPSILON = 1e-10
MIN_NORMALIZED_VALUE = 1e-8
MAX_NORMALIZED_VALUE = 0.999999
