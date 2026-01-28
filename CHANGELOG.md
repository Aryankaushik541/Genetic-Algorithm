# Changelog

All notable changes to the Genetic Algorithm Benchmark Suite.

## [2.0.0] - 2026-01-28

### 🎉 Major Release - Complete Overhaul

### ✨ Added

#### Core Features
- **Normalized Fitness Values**: All outputs now scaled to (0, 1) range, excluding zero
- **Interactive Menu System**: Two execution modes
  - Option 1: Individual function execution with dedicated table and graph
  - Option 2: Combined execution of all functions with comparative analysis
- **UV Package Manager Support**: Modern Python package management
  - Added `pyproject.toml` for UV configuration
  - Added `requirements.txt` for dependency management
- **Time-Limited Execution**: Maximum 20 seconds for all functions combined
- **Parallel Processing**: All functions run efficiently within time constraint

#### Documentation
- **Comprehensive README**: Complete installation and usage guide
- **Quick Start Guide**: Get started in 3 simple steps
- **MIT License**: Open source licensing
- **Changelog**: Track all project changes
- **.gitignore**: Proper Python project structure

#### Configuration
- **ga_config.py**: Centralized configuration file
  - Adjustable GA parameters
  - Execution time limits
  - Normalization parameters
  - Visualization settings

#### Visualization Improvements
- **Individual Function Plots**:
  - Convergence history graph
  - Distribution histogram with mean/median markers
- **Combined Analysis Plots**:
  - Best performance comparison (horizontal bar chart)
  - Average performance comparison (horizontal bar chart)
  - Top 5 functions convergence plot
  - Standard deviation analysis
- **Additional Plot Functions**:
  - Box plot comparison
  - Performance heatmap

### 🔧 Changed

#### Algorithm Improvements
- **Enhanced Normalization**: Logarithmic scaling for better value distribution
- **Improved GA Performance**: Optimized for faster execution
- **Better Statistics**: More robust statistical calculations
- **Adaptive Runs**: Automatically adjusts number of runs to meet time constraints

#### Code Structure
- **Modular Design**: Separated concerns into distinct modules
  - `ga_algorithm.py`: Core GA implementation
  - `benchmark_functions.py`: Test functions
  - `visualization.py`: All plotting functions
  - `ga_config.py`: Configuration parameters
  - `main_menu.py`: User interface
- **Better Documentation**: Comprehensive docstrings for all functions
- **Type Hints**: Improved code clarity

#### User Experience
- **Clear Output**: Better formatted tables and statistics
- **Progress Indicators**: Real-time feedback during execution
- **Error Handling**: Graceful error messages and recovery
- **Value Validation**: Ensures all outputs are in valid range

### 🐛 Fixed
- **Zero Values**: Eliminated zero values in normalized output
- **Execution Time**: Enforced 20-second maximum execution time
- **Function Compatibility**: All 15 functions work correctly
- **Visualization Scaling**: Proper axis limits and scaling

### 📊 Performance
- **Faster Execution**: Optimized algorithm for speed
- **Memory Efficient**: Better memory management
- **Scalable**: Handles all functions within time limit

### 🎯 Key Features Summary

1. **Normalized Values**: 0 < value < 1 (zero excluded)
2. **Two Execution Modes**:
   - Individual: Detailed analysis of single function
   - Combined: Comparative analysis of all functions
3. **Time Constraint**: Maximum 20 seconds execution
4. **UV Support**: Modern package management
5. **Comprehensive Docs**: README, Quick Start, and inline documentation
6. **Professional Visualization**: Multiple graph types
7. **Statistical Analysis**: Min, Mean, Median, Std Dev
8. **15 Benchmark Functions**: Standard optimization test suite

### 📦 Dependencies
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- tabulate >= 0.9.0

### 🚀 Installation

```bash
# Using UV (Recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

### 💻 Usage

```bash
python main_menu.py
```

### 🎓 Educational Value
- Learn Genetic Algorithms
- Understand optimization benchmarks
- Analyze algorithm performance
- Compare different test functions

### 🤝 Contributing
Contributions welcome! See README.md for guidelines.

---

## [1.0.0] - 2026-01-28

### Initial Release
- Basic GA implementation
- 15 benchmark functions
- Simple execution scripts
- Basic visualization

---

**Note**: This project follows [Semantic Versioning](https://semver.org/).
