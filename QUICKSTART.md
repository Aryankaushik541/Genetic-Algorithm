# 🚀 Quick Start Guide

Get started with the Genetic Algorithm Benchmark Suite in 3 simple steps!

## Step 1: Install UV (Recommended)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2: Setup Project

```bash
# Clone repository
git clone https://github.com/Aryankaushik541/Genetic-Algorithm.git
cd Genetic-Algorithm

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Step 3: Run the Program

```bash
python main_menu.py
```

## 📋 Menu Options

### Option 1: Individual Function
- Select a specific benchmark function
- View detailed statistics table
- See convergence graph
- Perfect for analyzing single function performance

### Option 2: All Functions Combined
- Runs all 15 functions in parallel
- Completes in ~20 seconds
- Shows comparative analysis
- Displays 4 comprehensive graphs

## 🎯 Example Usage

```
╔════════════════════════════════════════════════════════════╗
║     GENETIC ALGORITHM BENCHMARK SUITE                      ║
╚════════════════════════════════════════════════════════════╝

Select an option:
1. Run individual function (separate table & graph for each)
2. Run all functions together (combined table & graph)
3. Exit

Enter your choice (1-3): 2

🔄 Running all 15 functions...
   Time limit: 20 seconds
   Target runs per function: 30

Processing Sphere... Done
Processing Rastrigin... Done
...
```

## 📊 Understanding Results

All fitness values are **normalized** to range (0, 1):
- **Lower is better** (closer to 0)
- **Zero is excluded** (all values > 0)
- Values represent distance from optimal solution

### Metrics Explained:
- **Min**: Best performance across all runs
- **Mean**: Average performance
- **Median**: Middle value (robust to outliers)
- **Std Dev**: Consistency measure (lower = more consistent)

## 🔧 Customization

Edit `ga_config.py` to change parameters:

```python
NUM_DIMENSIONS = 5          # Problem size
POP_SIZE = 50              # Population size
NUM_GENERATIONS = 1000     # Evolution iterations
CROSSOVER_RATE = 0.8       # Crossover probability
MUTATION_RATE = 0.15       # Mutation probability
MAX_EXECUTION_TIME = 20    # Time limit (seconds)
```

## 💡 Tips

1. **For quick tests**: Use Option 1 with a single function
2. **For comprehensive analysis**: Use Option 2 for all functions
3. **Adjust time limit**: Modify `MAX_EXECUTION_TIME` in `ga_config.py`
4. **More runs**: Increase `NUM_RUNS` for better statistics (slower)
5. **Faster execution**: Decrease `NUM_GENERATIONS` or `POP_SIZE`

## 🐛 Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
uv pip install -r requirements.txt --force-reinstall
```

### Visualization Not Showing
```bash
# Install matplotlib backend
uv pip install PyQt5
# or
uv pip install tkinter
```

### Slow Execution
- Reduce `NUM_RUNS` in `ga_config.py`
- Reduce `NUM_GENERATIONS` for faster convergence
- Use Option 1 for single function testing

## 📚 Next Steps

- Read full [README.md](README.md) for detailed documentation
- Explore `benchmark_functions.py` to understand test functions
- Modify `ga_algorithm.py` to experiment with GA variants
- Check `visualization.py` for custom plotting options

## 🤝 Need Help?

- Check [README.md](README.md) for detailed information
- Review code comments in source files
- Open an issue on GitHub

Happy optimizing! 🎉
