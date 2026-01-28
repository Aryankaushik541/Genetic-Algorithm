"""
Configuration file for Genetic Algorithm parameters
Centralized settings for easy tuning
"""

# Genetic Algorithm Parameters
NUM_DIMENSIONS = 5          # Problem dimensionality
POP_SIZE = 50              # Population size
NUM_GENERATIONS = 1000     # Number of generations
CROSSOVER_RATE = 0.8       # Crossover probability
MUTATION_RATE = 0.15       # Mutation probability
ELITE_SIZE = 2             # Number of elite individuals to preserve

# Experiment Parameters
NUM_RUNS = 30              # Number of independent runs for individual function analysis
COMBINED_RUNS = 25         # Number of runs for combined analysis (all functions)
MAX_EXECUTION_TIME = 20    # Maximum execution time in seconds for all functions

# Normalization Parameters
EPSILON = 1e-10            # Small value to avoid division by zero
MIN_NORMALIZED_VALUE = 1e-8  # Minimum normalized value (excluding 0)
MAX_NORMALIZED_VALUE = 0.999999  # Maximum normalized value (excluding 1)

# Visualization Parameters
FIGURE_SIZE = (16, 12)     # Figure size for combined plots (width, height)
DPI = 100                  # Dots per inch for plots
PLOT_TOP_N_FUNCTIONS = 5   # Number of top functions to show in convergence plot

# Display Settings
TABLE_FORMAT = 'grid'      # Table format: 'grid', 'fancy_grid', 'simple', 'plain'
DECIMAL_PLACES = 6         # Number of decimal places to display

# Performance Thresholds (for interpretation)
EXCELLENT_THRESHOLD = 0.1   # Fitness < 0.1 is excellent
GOOD_THRESHOLD = 0.3        # Fitness < 0.3 is good
MODERATE_THRESHOLD = 0.5    # Fitness < 0.5 is moderate

# Consistency Thresholds (for std dev interpretation)
VERY_CONSISTENT = 0.01      # Std dev < 0.01 is very consistent
CONSISTENT = 0.05           # Std dev < 0.05 is consistent
