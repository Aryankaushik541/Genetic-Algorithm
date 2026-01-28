# 📤 Export Results Usage Guide

## Overview

The `export_results.py` utility allows you to run GA experiments and export results to CSV, JSON, and Markdown formats.

---

## 🚀 Quick Start

### Run All 15 Functions (Default)

```bash
# Export all 15 functions to all formats (CSV, JSON, MD)
python export_results.py all 30

# Export all 15 functions to CSV only
python export_results.py csv 20

# Export all 15 functions to JSON only
python export_results.py json 25

# Export all 15 functions to Markdown only
python export_results.py md 15
```

---

## 📋 Command Syntax

```bash
python export_results.py [selection] [format] [num_runs]
```

### Parameters:

1. **selection** (optional, default: 'all')
   - `all` - Run all 15 benchmark functions
   - `<function_name>` - Run specific function (e.g., 'Sphere', 'Rastrigin')

2. **format** (optional, default: 'all')
   - `all` - Export to CSV, JSON, and Markdown
   - `csv` - Export to CSV only
   - `json` - Export to JSON only
   - `md` - Export to Markdown only

3. **num_runs** (optional, default: 10)
   - Number of independent runs per function

---

## 📝 Usage Examples

### Example 1: All Functions, All Formats

```bash
python export_results.py all 30
```

**Output:**
- Runs all 15 functions
- 30 runs per function
- Exports to CSV, JSON, and Markdown
- Files: `ga_results_YYYYMMDD_HHMMSS.csv`, `.json`, `.md`

---

### Example 2: All Functions, CSV Only

```bash
python export_results.py csv 20
```

**Output:**
- Runs all 15 functions
- 20 runs per function
- Exports to CSV only
- File: `ga_results_YYYYMMDD_HHMMSS.csv`

---

### Example 3: Single Function, All Formats

```bash
python export_results.py Sphere all 10
```

**Output:**
- Runs Sphere function only
- 10 runs
- Exports to CSV, JSON, and Markdown
- Files: `ga_results_YYYYMMDD_HHMMSS.csv`, `.json`, `.md`

---

### Example 4: Single Function, JSON Only

```bash
python export_results.py Rastrigin json 15
```

**Output:**
- Runs Rastrigin function only
- 15 runs
- Exports to JSON only
- File: `ga_results_YYYYMMDD_HHMMSS.json`

---

## 📊 Output Formats

### 1. CSV Format

**File:** `ga_results_YYYYMMDD_HHMMSS.csv`

```csv
Function,Min,Mean,Median,Std Dev,Max,Runs
Sphere,0.001234,0.012345,0.010234,0.005678,0.023456,30
Rastrigin,0.234567,0.345678,0.323456,0.056789,0.456789,30
...
```

**Use Case:** Import into Excel, Google Sheets, or data analysis tools

---

### 2. JSON Format

**File:** `ga_results_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "2024-01-28T15:30:45.123456",
  "total_functions": 15,
  "results": {
    "Sphere": {
      "statistics": {
        "min": 0.001234,
        "mean": 0.012345,
        "median": 0.010234,
        "std": 0.005678,
        "max": 0.023456
      },
      "raw_values": [0.001234, 0.012345, ...],
      "num_runs": 30
    },
    ...
  }
}
```

**Use Case:** Import into Python, JavaScript, or web applications

---

### 3. Markdown Format

**File:** `ga_results_YYYYMMDD_HHMMSS.md`

```markdown
# Genetic Algorithm Results

**Generated:** 2024-01-28 15:30:45

**Total Functions:** 15

## Summary Table

| Rank | Function | Min | Mean | Median | Std Dev |
|------|----------|-----|------|--------|----------|
| 🥇 | Sphere | 0.001234 | 0.012345 | 0.010234 | 0.005678 |
| 🥈 | Sum_Diff_Powers | 0.002345 | 0.015678 | 0.013456 | 0.006789 |
| 🥉 | Schwefel_222 | 0.003456 | 0.018901 | 0.016789 | 0.007890 |
...

## Detailed Results

### Sphere

- **Best (Min):** 0.001234
- **Average (Mean):** 0.012345
- **Median:** 0.010234
- **Std Deviation:** 0.005678
- **Worst (Max):** 0.023456
- **Number of Runs:** 30

...

## Top 5 Performers

1. **Sphere** - Mean: 0.012345, Std: 0.005678
2. **Sum_Diff_Powers** - Mean: 0.015678, Std: 0.006789
...
```

**Use Case:** Documentation, reports, GitHub README

---

## 🎯 Common Use Cases

### Research Paper Data

```bash
# High statistical reliability
python export_results.py all 50
```

### Quick Testing

```bash
# Fast results
python export_results.py all 10
```

### Specific Function Analysis

```bash
# Deep dive into one function
python export_results.py Ackley all 100
```

### Comparison Study

```bash
# Compare specific functions
python export_results.py Sphere all 30
python export_results.py Rastrigin all 30
python export_results.py Ackley all 30
```

---

## 📁 Output Files

All files are timestamped to prevent overwriting:

```
ga_results_20240128_153045.csv
ga_results_20240128_153045.json
ga_results_20240128_153045.md
```

**Timestamp Format:** `YYYYMMDD_HHMMSS`

---

## 💡 Tips

### 1. **Choose Appropriate Run Count**

- **Quick Test:** 10 runs
- **Standard:** 20-30 runs
- **Research:** 50-100 runs

### 2. **Select Right Format**

- **CSV:** For spreadsheet analysis
- **JSON:** For programmatic access
- **Markdown:** For documentation
- **All:** When you need flexibility

### 3. **Batch Processing**

Run multiple exports with different parameters:

```bash
# Different run counts
python export_results.py all 10
python export_results.py all 30
python export_results.py all 50

# Different functions
python export_results.py Sphere all 30
python export_results.py Rastrigin all 30
python export_results.py Ackley all 30
```

---

## 🔍 Available Functions

All 15 benchmark functions:

1. Sphere
2. Rastrigin
3. Ackley
4. Griewank
5. Zakharov
6. Schwefel_222
7. Schwefel_12
8. Sum_Diff_Powers
9. Matyas
10. Dixon_Price
11. Levy
12. Perm
13. Rotated_Hyper_Ellipsoid
14. Bent_Cigar
15. Booth

---

## ⚡ Performance

### Execution Time Estimates:

| Functions | Runs | Estimated Time |
|-----------|------|----------------|
| 1 function | 10 runs | ~1 second |
| 1 function | 30 runs | ~3 seconds |
| All 15 | 10 runs | ~15 seconds |
| All 15 | 30 runs | ~45 seconds |

*Times are approximate and depend on system performance*

---

## 🐛 Troubleshooting

### Issue: "Function not found"

**Solution:** Check function name spelling (case-sensitive)

```bash
# ❌ Wrong
python export_results.py sphere all 10

# ✅ Correct
python export_results.py Sphere all 10
```

### Issue: "Invalid format"

**Solution:** Use valid format: 'all', 'csv', 'json', or 'md'

```bash
# ❌ Wrong
python export_results.py all txt 10

# ✅ Correct
python export_results.py all csv 10
```

---

## 📚 Integration Examples

### Python Script

```python
from export_results import run_and_export

# Run all functions
results = run_and_export('all', num_runs=30, export_format='all')

# Run specific function
results = run_and_export('Sphere', num_runs=50, export_format='json')

# Run multiple specific functions
results = run_and_export(['Sphere', 'Rastrigin'], num_runs=20, export_format='csv')
```

### Automated Testing

```bash
#!/bin/bash
# test_all.sh

echo "Running comprehensive tests..."

python export_results.py all 10
python export_results.py all 30
python export_results.py all 50

echo "All tests complete!"
```

---

## ✅ Summary

**Basic Usage:**
```bash
python export_results.py all 30
```

**Advanced Usage:**
```bash
python export_results.py <function> <format> <runs>
```

**Examples:**
- `python export_results.py all 30` - All functions, all formats
- `python export_results.py csv 20` - All functions, CSV only
- `python export_results.py Sphere all 10` - Sphere only, all formats
- `python export_results.py Rastrigin json 15` - Rastrigin only, JSON

---

**Happy Exporting! 📤🚀**
