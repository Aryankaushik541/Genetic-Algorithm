# 🧬 Genetic Algorithm Benchmark Suite

Simple, fast, and efficient Genetic Algorithm implementation for 15 benchmark functions with parallel execution.

## ✨ Features

- **15 Benchmark Functions** - Standard optimization test problems
- **⚡ Parallel Execution** - All functions run simultaneously (up to 15x faster!)
- **Simple Code** - Clean, minimal, easy to understand
- **Interactive Menu** - Run individual or all functions
- **Utilities** - Quick test, batch test, export (CSV/JSON/Markdown)
- **Visualization** - Convergence plots and performance comparisons

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Aryankaushik541/Genetic-Algorithm.git
cd Genetic-Algorithm

# Install
pip install -r requirements.txt

# Run
python main_menu.py
```

## 📊 Usage

### Interactive Menu

```bash
python main_menu.py
```

**Options:**
1. Run single function - Individual analysis with graphs
2. Run all functions - Parallel execution with comparison
3. Exit

### Utilities

```bash
# Quick test
python utils.py test Sphere 10

# Batch test
python utils.py batch Sphere Rastrigin Ackley 10

# Export all functions
python utils.py export all 30

# Export to CSV only
python utils.py export all 20 csv

# Export specific function
python utils.py export Sphere 10 json
```

## ⚡ Performance

| Mode | Time | Speedup |
|------|------|---------|
| Sequential | ~18s | 1x |
| Parallel (8 cores) | ~2.5s | **7x** |

All 15 functions × 25 runs = 375 total runs in ~2-5 seconds!

## 📁 Structure

```
Genetic-Algorithm/
├── main_menu.py           # Interactive menu (149 lines)
├── ga_algorithm.py        # Core GA (144 lines)
├── benchmark_functions.py # 15 functions (232 lines)
├── visualization.py       # Plotting (114 lines)
├── utils.py              # Utilities (193 lines)
├── config.py             # Settings (24 lines)
├── requirements.txt      # Dependencies
└── README.md            # This file

Total: ~850 lines of clean, simple code
```

## 🎯 Benchmark Functions

| Function | Type | Domain |
|----------|------|--------|
| Sphere | Unimodal | [-5.12, 5.12] |
| Rastrigin | Multimodal | [-5.12, 5.12] |
| Ackley | Multimodal | [-32.768, 32.768] |
| Griewank | Multimodal | [-600, 600] |
| Zakharov | Unimodal | [-5, 10] |
| Schwefel 2.22 | Unimodal | [-10, 10] |
| Schwefel 1.2 | Unimodal | [-100, 100] |
| Sum Diff Powers | Unimodal | [-1, 1] |
| Matyas | Unimodal | [-10, 10] |
| Dixon-Price | Unimodal | [-10, 10] |
| Levy | Multimodal | [-10, 10] |
| Perm | Multimodal | [-1, 1] |
| Rotated Hyper-Ellipsoid | Unimodal | [-65.536, 65.536] |
| Bent Cigar | Unimodal | [-100, 100] |
| Booth | Unimodal | [-10, 10] |

All functions have optimal value of 0.0

## ⚙️ Configuration

Edit `config.py`:

```python
# GA Parameters
NUM_DIMENSIONS = 5
POP_SIZE = 50
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_SIZE = 2

# Runs
NUM_RUNS = 30              # Individual
COMBINED_RUNS = 25         # All functions
```

## 📦 Dependencies

- numpy
- matplotlib
- tabulate

## 🌟 Highlights

✅ **Simple** - ~850 lines total, easy to understand  
✅ **Fast** - Parallel execution, 2-5 seconds for all functions  
✅ **Clean** - No unnecessary complexity  
✅ **Complete** - Menu, utilities, visualization, export  
✅ **Tested** - 15 standard benchmark functions  
✅ **Documented** - Clear code and README  

## 📄 License

MIT License - See LICENSE file

## 👨‍💻 Author

**Aryan Kaushik**

---

**Simple. Fast. Effective.** 🚀
