# 📊 Sample Output Examples

This document shows example outputs from the Genetic Algorithm Benchmark Suite.

---

## 🎮 Main Menu

```
============================================================
     GENETIC ALGORITHM BENCHMARK SUITE                      
============================================================
Normalized Fitness Values: 0 < value < 1 (excluding 0)
============================================================


Select an option:
1. Run individual function (separate table & graph for each)
2. Run all functions together (combined table & graph)
3. Exit

Enter your choice (1-3): 
```

---

## 📋 Option 1: Individual Function Execution

### Step 1: Function Selection

```
Enter your choice (1-3): 1

Available Benchmark Functions:
------------------------------------------------------------
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
------------------------------------------------------------

Enter function number (or 0 to go back): 1
```

### Step 2: Execution Progress

```
🔄 Running Sphere...
   Performing 30 independent runs...
```

### Step 3: Results Table

```
============================================================
                 RESULTS FOR SPHERE                         
============================================================
╒═════════════════╤═══════════╕
│ Metric          │ Value     │
╞═════════════════╪═══════════╡
│ Min             │ 0.123456  │
├─────────────────┼───────────┤
│ Mean            │ 0.234567  │
├─────────────────┼───────────┤
│ Median          │ 0.198765  │
├─────────────────┼───────────┤
│ Std Dev         │ 0.045678  │
├─────────────────┼───────────┤
│ Runs            │ 30        │
├─────────────────┼───────────┤
│ Execution Time  │ 3.45s     │
╘═════════════════╧═══════════╛

✓ All values are normalized: 0 < value < 1 (excluding 0)
✓ Value range: [0.123456, 0.298765]

📊 Generating visualization...
```

### Step 4: Individual Graphs

**Graph 1: Convergence History**
- X-axis: Generation (0-1000)
- Y-axis: Normalized Fitness (0-1)
- Shows how fitness improves over generations
- Text box with Final, Mean, Std Dev values

**Graph 2: Distribution of Results**
- Histogram showing distribution of 30 runs
- Red dashed line: Mean value
- Green dashed line: Median value
- X-axis: Normalized Fitness (0-1)
- Y-axis: Frequency

```
✓ Analysis complete!
```

---

## 🎯 Option 2: All Functions Combined

### Step 1: Execution Start

```
Enter your choice (1-3): 2

🔄 Running all 15 functions...
   Time limit: 20 seconds
   Target runs per function: 30

Running 30 runs per function to meet 20s time limit...
Processing Sphere... Done
Processing Rastrigin... Done
Processing Ackley... Done
Processing Griewank... Done
Processing Zakharov... Done
Processing Schwefel_222... Done
Processing Schwefel_12... Done
Processing Sum_Diff_Powers... Done
Processing Matyas... Done
Processing Dixon_Price... Done
Processing Levy... Done
Processing Perm... Done
Processing Rotated_Hyper_Ellipsoid... Done
Processing Bent_Cigar... Done
Processing Booth... Done
```

### Step 2: Combined Results Table

```
===============================================================================================
                         COMBINED RESULTS - ALL FUNCTIONS                                     
===============================================================================================
╒═══════════════════════════════╤════════════╤════════════╤════════════╤════════════╕
│ Function                      │ Min        │ Mean       │ Median     │ Std Dev    │
╞═══════════════════════════════╪════════════╪════════════╪════════════╪════════════╡
│ Sphere                        │ 0.001234   │ 0.012345   │ 0.010234   │ 0.005678   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Sum_Diff_Powers               │ 0.002345   │ 0.015678   │ 0.013456   │ 0.006789   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Schwefel_222                  │ 0.003456   │ 0.018901   │ 0.016789   │ 0.007890   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Zakharov                      │ 0.004567   │ 0.021234   │ 0.019012   │ 0.008901   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Schwefel_12                   │ 0.005678   │ 0.024567   │ 0.022345   │ 0.009012   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Rotated_Hyper_Ellipsoid       │ 0.006789   │ 0.027890   │ 0.025678   │ 0.010123   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Bent_Cigar                    │ 0.007890   │ 0.031234   │ 0.028901   │ 0.011234   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Dixon_Price                   │ 0.008901   │ 0.034567   │ 0.032234   │ 0.012345   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Matyas                        │ 0.009012   │ 0.037890   │ 0.035567   │ 0.013456   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Booth                         │ 0.010123   │ 0.041234   │ 0.038890   │ 0.014567   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Griewank                      │ 0.123456   │ 0.234567   │ 0.212345   │ 0.045678   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Ackley                        │ 0.234567   │ 0.345678   │ 0.323456   │ 0.056789   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Levy                          │ 0.345678   │ 0.456789   │ 0.434567   │ 0.067890   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Rastrigin                     │ 0.456789   │ 0.567890   │ 0.545678   │ 0.078901   │
├───────────────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Perm                          │ 0.567890   │ 0.678901   │ 0.656789   │ 0.089012   │
╘═══════════════════════════════╧════════════╧════════════╧════════════╧════════════╛

===============================================================================================
Total Execution Time: 18.45 seconds
Functions Evaluated: 15
Total Runs: 450
===============================================================================================

🏆 BEST PERFORMING FUNCTION: Sphere
   Mean Performance: 0.012345
   Standard Deviation: 0.005678

✓ All values are normalized: 0 < value < 1 (excluding 0)

📊 Generating combined visualization...
```

### Step 3: Combined Visualization (4 Graphs)

**Graph 1: Best Normalized Fitness by Function (Top Left)**
- Horizontal bar chart
- Functions sorted by best performance
- X-axis: Best Fitness (0-1)
- Value labels on bars
- Grid for easy reading

**Graph 2: Average Normalized Fitness by Function (Top Right)**
- Horizontal bar chart
- Functions sorted by average performance
- X-axis: Average Fitness (0-1)
- Value labels on bars
- Shows consistency across runs

**Graph 3: Convergence - Top 5 Functions (Bottom Left)**
- Line plot showing convergence over generations
- Top 5 best performing functions
- Best function has thicker line
- X-axis: Generation (0-1000)
- Y-axis: Normalized Fitness (0-1)
- Legend with function names
- Colorful lines (viridis colormap)

**Graph 4: Standard Deviation by Function (Bottom Right)**
- Horizontal bar chart
- Shows consistency/reliability
- Lower std dev = more consistent
- X-axis: Std Dev
- Value labels on bars

**Main Title:** "GA Performance Analysis - Normalized Values (0-1)"

```
✓ Analysis complete!
```

---

## 🔄 Return to Menu

```

Select an option:
1. Run individual function (separate table & graph for each)
2. Run all functions together (combined table & graph)
3. Exit

Enter your choice (1-3): 
```

---

## 👋 Exit Program

```
Enter your choice (1-3): 3

👋 Thank you for using GA Benchmark Suite!
============================================================
```

---

## 📈 Sample Detailed Output (Sphere Function)

### Convergence Pattern
```
Generation 0:    Fitness = 0.987654
Generation 100:  Fitness = 0.654321
Generation 200:  Fitness = 0.456789
Generation 300:  Fitness = 0.345678
Generation 400:  Fitness = 0.234567
Generation 500:  Fitness = 0.156789
Generation 600:  Fitness = 0.123456
Generation 700:  Fitness = 0.098765
Generation 800:  Fitness = 0.087654
Generation 900:  Fitness = 0.076543
Generation 1000: Fitness = 0.065432
```

### 30 Runs Results
```
Run  1: 0.123456 (seed=42)
Run  2: 0.134567 (seed=84)
Run  3: 0.145678 (seed=126)
Run  4: 0.156789 (seed=168)
Run  5: 0.167890 (seed=210)
...
Run 26: 0.287654 (seed=1092)
Run 27: 0.298765 (seed=1134)
Run 28: 0.309876 (seed=1176)
Run 29: 0.320987 (seed=1218)
Run 30: 0.332098 (seed=1260)
```

---

## 🎨 Visual Elements Description

### Individual Function Visualization
```
┌─────────────────────────────────────────────────────────────┐
│  Sphere - Convergence History    │  Sphere - Distribution   │
│                                   │                          │
│  1.0 ┐                           │   8 ┐                    │
│      │╲                          │     │    ┌─┐             │
│  0.8 │ ╲                         │   6 │  ┌─┤ ├─┐           │
│      │  ╲___                     │     │  │ │ │ │           │
│  0.6 │      ╲___                 │   4 │┌─┤ │ │ ├─┐         │
│      │          ╲___             │     ││ │ │ │ │ │         │
│  0.4 │              ╲___         │   2 ││ │ │ │ │ │         │
│      │                  ╲___     │     ││ │ │ │ │ │         │
│  0.2 │                      ╲___ │   0 └┴─┴─┴─┴─┴─┴─        │
│      │                          ╲│     0.0  0.2  0.4  0.6   │
│  0.0 └──────────────────────────┘│                          │
│      0    250   500   750  1000  │  Mean: 0.234 (red line) │
│      Generation                   │  Median: 0.198 (green)  │
└─────────────────────────────────────────────────────────────┘
```

### Combined Visualization Layout
```
┌──────────────────────────────────────────────────────────────┐
│        GA Performance Analysis - Normalized Values (0-1)     │
├──────────────────────────┬───────────────────────────────────┤
│ Best Fitness by Function │ Avg Fitness by Function           │
│                          │                                   │
│ Sphere        ▓░░░       │ Sphere        ▓▓░░░              │
│ Sum_Diff      ▓░░░       │ Sum_Diff      ▓▓░░░              │
│ Schwefel_222  ▓░░░       │ Schwefel_222  ▓▓░░░              │
│ ...                      │ ...                               │
├──────────────────────────┼───────────────────────────────────┤
│ Convergence - Top 5      │ Standard Deviation                │
│                          │                                   │
│ 1.0 ┐ ─── Sphere        │ Sphere        ▓░░                 │
│     │ ─── Sum_Diff      │ Sum_Diff      ▓░░                 │
│ 0.8 │ ─── Schwefel      │ Schwefel_222  ▓░░                 │
│     │ ─── Zakharov      │ Zakharov      ▓░░                 │
│ 0.6 │ ─── Schwefel_12   │ Schwefel_12   ▓░░                 │
│     │                    │ ...                               │
│ 0.4 │                    │                                   │
│     │                    │                                   │
│ 0.2 │                    │                                   │
│     │                    │                                   │
│ 0.0 └────────────────    │                                   │
│     0   500   1000       │                                   │
└──────────────────────────┴───────────────────────────────────┘
```

---

## 📊 Key Observations from Output

### ✅ Value Validation
- All values strictly between 0 and 1
- No zero values present
- Proper normalization applied

### 📈 Performance Insights
- **Unimodal functions** (Sphere, Zakharov) → Lower fitness values
- **Multimodal functions** (Rastrigin, Levy) → Higher fitness values
- **Consistency** shown by standard deviation

### ⚡ Execution Efficiency
- Individual function: ~3-5 seconds
- All functions combined: ~18-20 seconds (within limit)
- 30 runs per function for statistical reliability

### 🎯 Statistical Reliability
- Different seeds (i*42) ensure independence
- 30 runs provide robust statistics
- Min, Mean, Median, Std Dev give complete picture

---

## 💡 Interpretation Guide

### Fitness Values
- **0.001 - 0.100**: Excellent performance
- **0.100 - 0.300**: Good performance
- **0.300 - 0.500**: Moderate performance
- **0.500 - 0.700**: Poor performance
- **0.700 - 1.000**: Very poor performance

### Standard Deviation
- **< 0.010**: Very consistent
- **0.010 - 0.050**: Consistent
- **0.050 - 0.100**: Moderate variance
- **> 0.100**: High variance (less reliable)

---

**Note:** Actual values will vary based on random seeds and GA performance. This is a representative example showing the structure and format of outputs.
