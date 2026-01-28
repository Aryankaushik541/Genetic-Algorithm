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
- **Utility Scripts**: Quick testing, batch analysis, and results export tools

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
2. Run all functions together (combined table & graph)
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

# Export results to CSV/JSON/Markdown
python export_results.py all 30
```

See [UTILITIES.md](UTILITIES.md) for complete utility documentation.

## 📊 Output Examples

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
├── 📖 Documentation
│   ├── README.md              # This file
│   ├── QUICKSTART.md          # Quick setup guide
│   ├── SAMPLE_OUTPUT.md       # Output examples
│   ├── DEMO.md                # Visual demo
│   ├── UTILITIES.md           # Utility scripts guide
│   ├── CHANGELOG.md           # Version history
│   └── LICENSE                # MIT License
│
├── 🎮 Main Application
│   ├── main_menu.py           # Interactive menu system
│   ├── ga_algorithm.py        # Core GA implementation
│   ├── benchmark_functions.py # 15 benchmark functions
│   ├── visualization.py       # Graph generation
│   └── results_table.py       # Table display utilities
│
├── 🛠️ Utility Scripts
│   ├── quick_test.py          # Fast function testing
│   ├── batch_analysis.py      # Batch testing & comparison
│   └── export_results.py      # Results export (CSV/JSON/MD)
│
├── 🔧 Configuration
│   ├── ga_config.py           # GA parameters
│   ├── requirements.txt       # Python dependencies
│   ├── pyproject.toml         # UV configuration
│   └── .gitignore            # Git ignore rules
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

### Quick Test
```bash
python quick_test.py Sphere 10
```
Fast testing of individual functions without full menu.

### Batch Analysis
```python
from batch_analysis import batch_test_functions
results = batch_test_functions(['Sphere', 'Rastrigin'], num_runs=10)
```
Automated batch testing and configuration comparison.

### Export Results
```bash
python export_results.py all 30
```
Export results to CSV, JSON, and Markdown formats.

See [UTILITIES.md](UTILITIES.md) for complete documentation.

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 3 simple steps
- **[SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md)** - See example outputs
- **[DEMO.md](DEMO.md)** - Visual walkthrough with ASCII art
- **[UTILITIES.md](UTILITIES.md)** - Utility scripts guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Aryan Kaushik**

## 🙏 Acknowledgments

- Benchmark functions based on standard optimization test problems
- Genetic Algorithm implementation inspired by classical GA literature

## 🌟 Key Highlights

✅ **Normalized Values**: 0 < value < 1 (zero excluded)  
✅ **Fast Execution**: 20 seconds max for all functions  
✅ **Statistical Rigor**: 30 independent runs with i*seed randomization  
✅ **Professional Visualization**: Multiple graph types  
✅ **Comprehensive Documentation**: README, Quick Start, Demo, Utilities  
✅ **Utility Scripts**: Quick testing, batch analysis, export tools  
✅ **UV Support**: Modern Python package management  
✅ **Production Ready**: Clean code, proper structure, MIT licensed
