# 🧬 Genetic Algorithm Benchmark Suite

A comprehensive Genetic Algorithm implementation for optimizing 15 benchmark functions with advanced visualization and statistical analysis.

## 📋 Features

- **15 Benchmark Functions**: Sphere, Rastrigin, Ackley, Griewank, Zakharov, Schwefel 2.22, Schwefel 1.2, Sum of Different Powers, Matyas, Dixon-Price, Levy, Perm, Rotated Hyper-Ellipsoid, Bent Cigar, and Booth
- **Normalized Fitness Values**: All outputs scaled between 0 and 1 (excluding zero)
- **Interactive Menu System**: 
  - Option 1: Run individual functions with separate tables and graphs
  - Option 2: Run all functions together with combined visualization
- **Fast Execution**: Maximum 20 seconds execution time with parallel processing
- **Statistical Analysis**: Min, Mean, Median, and Standard Deviation for 30 runs
- **Advanced Visualization**: Convergence plots, performance comparisons, and distribution analysis

## 🚀 Installation

### Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/Aryankaushik541/Genetic-Algorithm.git
cd Genetic-Algorithm

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/Aryankaushik541/Genetic-Algorithm.git
cd Genetic-Algorithm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- numpy
- matplotlib
- tabulate

## 🎮 Usage

### Interactive Menu

Run the main program:

```bash
python main_menu.py
```

You'll see:

```
╔════════════════════════════════════════════════════════════╗
║     GENETIC ALGORITHM BENCHMARK SUITE                      ║
╚════════════════════════════════════════════════════════════╝

Select an option:
1. Run individual function (separate table & graph for each)
2. Run all functions together (combined table & graph)
3. Exit

Enter your choice (1-3):
```

### Option 1: Individual Function Execution

- Select a specific function from the list
- View detailed statistics table for that function
- See convergence graph for that function only
- Results are normalized between 0 and 1

### Option 2: Combined Execution

- Runs all 15 functions in parallel
- Completes within 20 seconds
- Shows comparative table for all functions
- Displays combined visualization with 4 graphs:
  - Best performance comparison
  - Average performance comparison
  - Convergence of top 5 functions
  - Standard deviation analysis

## 📊 Output Examples

### Individual Function Table

```
📊 Sphere Function
╒═════════╤═══════════╕
│ Metric  │ Value     │
╞═════════╪═══════════╡
│ Min     │ 0.123456  │
│ Mean    │ 0.234567  │
│ Median  │ 0.198765  │
│ Std Dev │ 0.045678  │
╘═════════╧═══════════╛
```

### Combined Comparison Table

```
╒═══════════════════════════╤═══════════╤═══════════╤═══════════╤═══════════╕
│ Function                  │ Min       │ Mean      │ Median    │ Std Dev   │
╞═══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ Sphere                    │ 0.123456  │ 0.234567  │ 0.198765  │ 0.045678  │
│ Rastrigin                 │ 0.345678  │ 0.456789  │ 0.412345  │ 0.067890  │
│ ...                       │ ...       │ ...       │ ...       │ ...       │
╘═══════════════════════════╧═══════════╧═══════════╧═══════════╧═══════════╛
```

## 🔧 Configuration

Edit `ga_config.py` to modify GA parameters:

```python
NUM_DIMENSIONS = 5          # Problem dimensionality
POP_SIZE = 50              # Population size
NUM_GENERATIONS = 1000     # Number of generations
CROSSOVER_RATE = 0.8       # Crossover probability
MUTATION_RATE = 0.15       # Mutation probability
ELITE_SIZE = 2             # Number of elite individuals
MAX_EXECUTION_TIME = 20    # Maximum execution time in seconds
```

## 📁 Project Structure

```
Genetic-Algorithm/
├── main_menu.py              # Interactive menu system
├── ga_algorithm.py           # Core GA implementation
├── benchmark_functions.py    # 15 benchmark functions
├── results_table.py          # Table display utilities
├── visualization.py          # Graph generation
├── ga_config.py             # Configuration parameters
├── requirements.txt         # Python dependencies
├── pyproject.toml          # UV configuration
└── README.md               # This file
```

## 🎯 Benchmark Functions

| Function | Optimal Value | Search Domain |
|----------|--------------|---------------|
| Sphere | 0.0 | [-5.12, 5.12] |
| Rastrigin | 0.0 | [-5.12, 5.12] |
| Ackley | 0.0 | [-32.768, 32.768] |
| Griewank | 0.0 | [-600, 600] |
| Zakharov | 0.0 | [-5, 10] |
| Schwefel 2.22 | 0.0 | [-10, 10] |
| Schwefel 1.2 | 0.0 | [-100, 100] |
| Sum Diff Powers | 0.0 | [-1, 1] |
| Matyas | 0.0 | [-10, 10] |
| Dixon-Price | 0.0 | [-10, 10] |
| Levy | 0.0 | [-10, 10] |
| Perm | 0.0 | [-1, 1] |
| Rotated Hyper-Ellipsoid | 0.0 | [-65.536, 65.536] |
| Bent Cigar | 0.0 | [-100, 100] |
| Booth | 0.0 | [-10, 10] |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Aryan Kaushik**

## 🙏 Acknowledgments

- Benchmark functions based on standard optimization test problems
- Genetic Algorithm implementation inspired by classical GA literature
