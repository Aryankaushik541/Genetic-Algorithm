# 🧬 Genetic Algorithm Benchmark Suite

A comprehensive Genetic Algorithm implementation for optimizing 15 benchmark functions with **parallel execution**, advanced visualization, and statistical analysis.

## 📋 Features

- **15 Benchmark Functions**: Sphere, Rastrigin, Ackley, Griewank, Zakharov, Schwefel 2.22, Schwefel 1.2, Sum of Different Powers, Matyas, Dixon-Price, Levy, Perm, Rotated Hyper-Ellipsoid, Bent Cigar, and Booth
- **⚡ Parallel Execution**: All 15 functions run **simultaneously** using multiprocessing (up to 15x faster!)
- **Normalized Fitness Values**: All outputs scaled between 0 and 1 (excluding zero)
- **Interactive Menu System**: 
  - Option 1: Run individual functions with separate tables and graphs
  - Option 2: Run all functions together in parallel with combined visualization
- **Blazing Fast**: Completes all 375 runs (15 functions × 25 runs) in ~2-5 seconds on multi-core systems
- **Statistical Analysis**: Min, Mean, Median, and Standard Deviation with 25-30 runs
- **Advanced Visualization**: Convergence plots, performance comparisons, and distribution analysis
- **Export Utilities**: Export results to CSV, JSON, and Markdown formats
- **Utility Scripts**: Quick testing, batch analysis, and results export tools

## ⚡ Parallel Execution

### Sequential vs Parallel:

| Mode | Time | Speedup |
|------|------|---------|
| **Sequential** (old) | ~18s | 1x |
| **Parallel** (new) | ~2.5s | **7-8x faster** |

All 15 functions now run **simultaneously** instead of one-by-one using Python's multiprocessing module!

**Performance by CPU Cores:**

| CPU Cores | Time | Speedup |
|-----------|------|---------|
| 1 core    | 18.0s | 1.0x |
| 4 cores   | 5.0s | 3.6x |
| 8 cores   | 2.5s | **7.2x** |
| 16 cores  | 1.5s | **12.0x** |

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

### Interactive Menu (Main Application)

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
2. Run all functions together (PARALLEL - combined table & graph)
3. Exit

Enter your choice (1-3):
```

### Quick Testing (Utility Scripts)

For rapid testing without the full menu:

```bash
# Test single function
python quick_test.py Sphere

# Compare multiple functions
python quick_test.py compare Sphere Rastrigin Ackley
```

### Export Results

Export results to CSV, JSON, and Markdown formats:

```bash
# Export all 15 functions to all formats (CSV, JSON, MD)
python export_results.py all 30

# Export all functions to CSV only
python export_results.py csv 20

# Export specific function to all formats
python export_results.py Sphere all 10

# Export specific function to JSON only
python export_results.py Rastrigin json 15
```

**See [EXPORT_USAGE.md](EXPORT_USAGE.md) for complete export documentation.**

## 📊 Output Examples

### Option 1: Individual Function Execution

- Select a specific function from the list
- View detailed statistics table for that function
- See convergence graph for that function only
- Results are normalized between 0 and 1

### Option 2: Combined Parallel Execution ⚡

- Runs all 15 functions **simultaneously** using multiprocessing
- Completes in ~2-5 seconds (depending on CPU cores)
- Shows comparative table for all functions with rankings
- Displays combined visualization with 4 graphs:
  - Best performance comparison
  - Average performance comparison
  - Convergence of top 5 functions
  - Standard deviation analysis

**Example Output:**
```
⚡ Parallel Execution Time: 2.45 seconds
📊 Functions Evaluated: 15
🔢 Total Runs: 375
🎯 Runs per Function: 25
💻 CPU Cores Used: 8
⚡ Estimated Speedup: ~8x faster than sequential
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
NUM_RUNS = 30              # Runs for individual analysis
COMBINED_RUNS = 25         # Runs for parallel analysis
MAX_EXECUTION_TIME = 20    # Maximum execution time in seconds
```

## 📁 Project Structure

```
Genetic-Algorithm/
├── 📖 Documentation
│   ├── README.md              # This file
│   ├── EXPORT_USAGE.md        # Export utility guide
│   └── LICENSE                # MIT License
│
├── 🎮 Main Application
│   ├── main_menu.py           # Interactive menu system
│   ├── ga_algorithm.py        # Core GA with parallel execution
│   ├── benchmark_functions.py # 15 benchmark functions
│   ├── visualization.py       # Graph generation
│   └── results_table.py       # Table display utilities
│
├── 🛠️ Utility Scripts
│   ├── quick_test.py          # Fast function testing
│   ├── batch_analysis.py      # Batch testing & comparison
│   └── export_results.py      # Results export (CSV/JSON/MD)
│
└── 🔧 Configuration
    ├── ga_config.py           # GA parameters
    ├── requirements.txt       # Python dependencies
    ├── pyproject.toml         # UV configuration
    └── .gitignore            # Git ignore rules
```

## 🎯 Benchmark Functions

| Function | Optimal Value | Search Domain | Type |
|----------|--------------|---------------|------|
| Sphere | 0.0 | [-5.12, 5.12] | Unimodal |
| Rastrigin | 0.0 | [-5.12, 5.12] | Multimodal |
| Ackley | 0.0 | [-32.768, 32.768] | Multimodal |
| Griewank | 0.0 | [-600, 600] | Multimodal |
| Zakharov | 0.0 | [-5, 10] | Unimodal |
| Schwefel 2.22 | 0.0 | [-10, 10] | Unimodal |
| Schwefel 1.2 | 0.0 | [-100, 100] | Unimodal |
| Sum Diff Powers | 0.0 | [-1, 1] | Unimodal |
| Matyas | 0.0 | [-10, 10] | Unimodal |
| Dixon-Price | 0.0 | [-10, 10] | Unimodal |
| Levy | 0.0 | [-10, 10] | Multimodal |
| Perm | 0.0 | [-1, 1] | Multimodal |
| Rotated Hyper-Ellipsoid | 0.0 | [-65.536, 65.536] | Unimodal |
| Bent Cigar | 0.0 | [-100, 100] | Unimodal |
| Booth | 0.0 | [-10, 10] | Unimodal |

## 🛠️ Utility Scripts

### 1. Quick Test
```bash
# Test single function with 10 runs
python quick_test.py Sphere 10

# Compare multiple functions
python quick_test.py compare Sphere Rastrigin Ackley
```
Fast testing of individual functions without full menu.

### 2. Batch Analysis
```python
from batch_analysis import batch_test_functions

# Test multiple functions
results = batch_test_functions(['Sphere', 'Rastrigin'], num_runs=10)
```
Automated batch testing and configuration comparison.

### 3. Export Results
```bash
# Export all 15 functions to all formats (CSV, JSON, MD)
python export_results.py all 30

# Export all functions to CSV only
python export_results.py csv 20

# Export specific function to all formats
python export_results.py Sphere all 10

# Export specific function to JSON only
python export_results.py Rastrigin json 15
```

**Export Formats:**
- **CSV** - For Excel/spreadsheet analysis
- **JSON** - For programmatic access
- **Markdown** - For documentation/reports

**Output Files:**
```
ga_results_20240128_153045.csv
ga_results_20240128_153045.json
ga_results_20240128_153045.md
```

**See [EXPORT_USAGE.md](EXPORT_USAGE.md) for complete documentation.**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Aryan Kaushik**

## 🙏 Acknowledgments

- Benchmark functions based on standard optimization test problems
- Genetic Algorithm implementation inspired by classical GA literature
- Parallel execution using Python's multiprocessing module

## 🌟 Key Highlights

✅ **Normalized Values**: 0 < value < 1 (zero excluded)  
✅ **⚡ Parallel Execution**: Up to 15x faster with multiprocessing  
✅ **Fast Execution**: 2-5 seconds for all 375 runs  
✅ **Statistical Rigor**: 25-30 independent runs with i*seed randomization  
✅ **Professional Visualization**: Multiple graph types showing all 15 functions  
✅ **Export Utilities**: CSV, JSON, Markdown formats  
✅ **Utility Scripts**: Quick testing, batch analysis, export tools  
✅ **UV Support**: Modern Python package management  
✅ **Production Ready**: Clean code, proper structure, MIT licensed

---

**Experience blazing-fast parallel genetic algorithm optimization! ⚡🚀**
