# 🛠️ Utility Scripts Guide

Additional utility scripts for advanced usage and automation.

---

## 📋 Available Utilities

### 1. **quick_test.py** - Fast Function Testing

Quick testing of individual functions without the full menu system.

#### Usage:

```bash
# Test single function (default 10 runs)
python quick_test.py Sphere

# Test with custom number of runs
python quick_test.py Rastrigin 20

# List all available functions
python quick_test.py list

# Compare multiple functions
python quick_test.py compare Sphere Rastrigin Ackley
```

#### Examples:

**Single Function Test:**
```bash
$ python quick_test.py Sphere

============================================================
                 QUICK TEST: SPHERE                         
============================================================

Running 10 independent experiments...

✓ Completed in 1.23 seconds

╒═════════════════╤═══════════╕
│ Metric          │ Value     │
╞═════════════════╪═══════════╡
│ Best (Min)      │ 0.123456  │
│ Average (Mean)  │ 0.234567  │
│ Median          │ 0.198765  │
│ Std Deviation   │ 0.045678  │
│ Worst (Max)     │ 0.332098  │
│ Runs            │ 10        │
│ Time            │ 1.23s     │
╘═════════════════╧═══════════╛

============================================================
🏆 Performance: EXCELLENT
🎯 Consistency: VERY HIGH
============================================================
```

**Compare Functions:**
```bash
$ python quick_test.py compare Sphere Rastrigin Ackley

Testing Sphere... ✓ Mean: 0.123456
Testing Rastrigin... ✓ Mean: 0.456789
Testing Ackley... ✓ Mean: 0.234567

╒═══════╤═══════════╤═══════════╤═══════════╕
│ Rank  │ Function  │ Mean      │ Std Dev   │
╞═══════╪═══════════╪═══════════╪═══════════╡
│ 🥇    │ Sphere    │ 0.123456  │ 0.045678  │
│ 🥈    │ Ackley    │ 0.234567  │ 0.056789  │
│ 🥉    │ Rastrigin │ 0.456789  │ 0.078901  │
╘═══════╧═══════════╧═══════════╧═══════════╛

🏆 Best: Sphere (Mean: 0.123456)
```

---

### 2. **batch_analysis.py** - Automated Batch Testing

Advanced batch testing and configuration comparison.

#### Features:

- **Configuration Comparison**: Test different GA parameters
- **Batch Testing**: Test multiple functions quickly
- **Sensitivity Analysis**: Analyze parameter sensitivity
- **CSV Export**: Export results for further analysis

#### Usage:

```python
from batch_analysis import batch_test_functions, compare_configurations

# Quick batch test
test_functions = ['Sphere', 'Rastrigin', 'Ackley']
results = batch_test_functions(test_functions, num_runs=10)

# Compare configurations
configs = [
    {'pop_size': 50, 'mutation_rate': 0.1},
    {'pop_size': 100, 'mutation_rate': 0.15},
    {'pop_size': 50, 'mutation_rate': 0.2}
]
compare_configurations('Sphere', configs, num_runs=10)
```

#### Example Output:

```
════════════════════════════════════════════════════════════
              BATCH TESTING 3 FUNCTIONS                     
════════════════════════════════════════════════════════════

Testing Sphere... Done (1.2s)
Testing Rastrigin... Done (1.3s)
Testing Ackley... Done (1.2s)

════════════════════════════════════════════════════════════
                  BATCH TEST RESULTS                        
════════════════════════════════════════════════════════════

╒═══════════╤═══════════╤═══════════╤═══════════╤═══════╕
│ Function  │ Min       │ Mean      │ Std Dev   │ Time  │
╞═══════════╪═══════════╪═══════════╪═══════════╪═══════╡
│ Sphere    │ 0.123456  │ 0.234567  │ 0.045678  │ 1.2s  │
│ Rastrigin │ 0.456789  │ 0.567890  │ 0.078901  │ 1.3s  │
│ Ackley    │ 0.234567  │ 0.345678  │ 0.056789  │ 1.2s  │
╘═══════════╧═══════════╧═══════════╧═══════════╧═══════╛

Total Execution Time: 3.70 seconds
Average Time per Function: 1.23 seconds
```

---

### 3. **export_results.py** - Results Export

Export GA results to multiple formats for documentation and analysis.

#### Supported Formats:

- **CSV**: Spreadsheet-compatible format
- **JSON**: Structured data format
- **Markdown**: Documentation-ready format

#### Usage:

```bash
# Export to all formats (CSV, JSON, Markdown)
python export_results.py all 30

# Export to CSV only
python export_results.py csv 20

# Export to JSON only
python export_results.py json 10

# Export to Markdown only
python export_results.py md 15
```

#### Programmatic Usage:

```python
from export_results import run_and_export, export_all_formats

# Run experiments and export
results = run_and_export(
    function_names=['Sphere', 'Rastrigin'],
    num_runs=30,
    export_format='all'
)

# Or export existing results
export_all_formats(results_dict, base_filename='my_results')
```

#### Output Files:

**CSV Format (`ga_results_20260128_143022.csv`):**
```csv
Function,Min,Mean,Median,Std Dev,Max,Runs
Sphere,0.123456,0.234567,0.198765,0.045678,0.332098,30
Rastrigin,0.456789,0.567890,0.545678,0.078901,0.689012,30
```

**JSON Format (`ga_results_20260128_143022.json`):**
```json
{
  "timestamp": "2026-01-28T14:30:22",
  "total_functions": 2,
  "results": {
    "Sphere": {
      "statistics": {
        "min": 0.123456,
        "mean": 0.234567,
        "median": 0.198765,
        "std": 0.045678,
        "max": 0.332098
      },
      "raw_values": [0.123456, 0.134567, ...],
      "num_runs": 30
    }
  }
}
```

**Markdown Format (`ga_results_20260128_143022.md`):**
```markdown
# Genetic Algorithm Results

**Generated:** 2026-01-28 14:30:22
**Total Functions:** 2

## Summary Table

| Rank | Function | Min | Mean | Median | Std Dev |
|------|----------|-----|------|--------|----------|
| 🥇 | Sphere | 0.123456 | 0.234567 | 0.198765 | 0.045678 |
| 🥈 | Rastrigin | 0.456789 | 0.567890 | 0.545678 | 0.078901 |
```

---

## 🎯 Use Cases

### Research & Documentation

```bash
# Run comprehensive analysis and export
python export_results.py all 30

# Generate publication-ready tables
python export_results.py md 50
```

### Quick Testing During Development

```bash
# Test single function quickly
python quick_test.py Sphere 5

# Compare different functions
python quick_test.py compare Sphere Rastrigin Ackley
```

### Parameter Tuning

```python
from batch_analysis import sensitivity_analysis

# Test different mutation rates
sensitivity_analysis(
    'Sphere',
    'mutation_rate',
    [0.05, 0.1, 0.15, 0.2, 0.25],
    num_runs=10
)
```

### Automated Testing

```python
from batch_analysis import batch_test_functions

# Test all functions quickly
results = batch_test_functions(num_runs=5)

# Test specific subset
test_set = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank']
results = batch_test_functions(test_set, num_runs=10)
```

---

## 📊 Comparison: Main Menu vs Utilities

| Feature | main_menu.py | Utilities |
|---------|--------------|-----------|
| **Interactive** | ✅ Full menu | ❌ Command-line |
| **Visualization** | ✅ Graphs | ❌ Tables only |
| **Speed** | Moderate | ⚡ Fast |
| **Export** | ❌ No | ✅ CSV/JSON/MD |
| **Automation** | ❌ Manual | ✅ Scriptable |
| **Best For** | Analysis & Visualization | Quick tests & Automation |

---

## 💡 Tips

1. **Quick Testing**: Use `quick_test.py` for rapid function evaluation
2. **Batch Processing**: Use `batch_analysis.py` for testing multiple configurations
3. **Documentation**: Use `export_results.py` to generate reports
4. **Automation**: Combine utilities in shell scripts for automated testing

---

## 🔧 Advanced Examples

### Automated Nightly Testing

```bash
#!/bin/bash
# nightly_test.sh

echo "Running nightly GA tests..."

# Quick test all functions
python quick_test.py compare Sphere Rastrigin Ackley Griewank Zakharov

# Export detailed results
python export_results.py all 50

echo "Tests complete! Results exported."
```

### Parameter Sweep

```python
# parameter_sweep.py
from batch_analysis import sensitivity_analysis

functions = ['Sphere', 'Rastrigin', 'Ackley']
mutation_rates = [0.05, 0.1, 0.15, 0.2, 0.25]

for func in functions:
    print(f"\nAnalyzing {func}...")
    sensitivity_analysis(func, 'mutation_rate', mutation_rates, num_runs=10)
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: GA Tests

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run quick tests
        run: |
          python quick_test.py Sphere 10
          python quick_test.py Rastrigin 10
```

---

## 📚 Related Documentation

- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md) - Output examples
- [DEMO.md](DEMO.md) - Visual demo

---

**Happy Testing! 🚀**
