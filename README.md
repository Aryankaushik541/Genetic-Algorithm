# 🧬 Genetic Algorithm Benchmark Suite

Simple and effective GA for 15 benchmark functions with excellent results.

## ✨ Features

- **15 Benchmark Functions** - Standard test problems
- **⚡ Parallel Execution** - All functions run together (fast!)
- **Simple Code** - Easy to understand
- **Good Results** - Optimized for convergence
- **Interactive Menu** - Easy to use
- **Visualization** - Graphs and comparisons

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Aryankaushik541/Genetic-Algorithm.git
cd Genetic-Algorithm

# Install (if pip blocked: python -m pip install ...)
pip install numpy matplotlib tabulate

# Run
python main_menu.py
```

## 📊 Usage

### Main Menu
```bash
python main_menu.py

1. Run single function
2. Run all functions
3. Exit
```

### Utilities
```bash
# Test one function
python utils.py test Sphere 10

# Test multiple
python utils.py batch Sphere Rastrigin Ackley 10

# Export results
python utils.py export all 30
```

## 🎯 Benchmark Functions

All 15 functions have optimal value = 0.0

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

## ⚙️ Configuration

Edit `config.py`:

```python
NUM_DIMENSIONS = 5
POP_SIZE = 100
NUM_GENERATIONS = 1500
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.15
ELITE_SIZE = 3
```

## 📁 Files

```
main_menu.py           - Interactive menu
ga_algorithm.py        - GA implementation
benchmark_functions.py - 15 test functions
visualization.py       - Graphs
utils.py              - Utilities
config.py             - Settings
```

## 📦 Dependencies

```
numpy
matplotlib
tabulate
```

## 🔧 If pip is blocked

```bash
python -m pip install numpy matplotlib tabulate
```

## 📄 License

MIT License

## 👨‍💻 Author

Aryan Kaushik

---

**Simple. Fast. Good Results.** 🚀
