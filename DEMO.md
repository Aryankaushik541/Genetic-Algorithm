# 🎬 Visual Demo Guide

Complete walkthrough with visual representations of the Genetic Algorithm Benchmark Suite.

---

## 🚀 Starting the Program

```bash
$ python main_menu.py
```

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     🧬 GENETIC ALGORITHM BENCHMARK SUITE 🧬                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

📊 Normalized Fitness Values: 0 < value < 1 (excluding 0)
🎯 15 Benchmark Functions Available
⚡ Fast Execution (Max 20 seconds for all functions)
📈 Statistical Analysis with 30 Independent Runs

════════════════════════════════════════════════════════════

Select an option:

  [1] 🔍 Run Individual Function
      → Detailed analysis of single function
      → Separate table and convergence graph
      → Distribution histogram
      
  [2] 📊 Run All Functions Combined
      → Comparative analysis of all 15 functions
      → Combined results table
      → 4 comprehensive graphs
      → Completes in ~20 seconds
      
  [3] 🚪 Exit

════════════════════════════════════════════════════════════

Enter your choice (1-3): _
```

---

## 🔍 Demo 1: Individual Function Analysis

### Step 1: Select Function

```
Enter your choice (1-3): 1

╔════════════════════════════════════════════════════════════╗
║           AVAILABLE BENCHMARK FUNCTIONS                    ║
╚════════════════════════════════════════════════════════════╝

┌────┬─────────────────────────────┬──────────────┬───────────┐
│ #  │ Function Name               │ Type         │ Dimension │
├────┼─────────────────────────────┼──────────────┼───────────┤
│  1 │ Sphere                      │ Unimodal     │ Any       │
│  2 │ Rastrigin                   │ Multimodal   │ Any       │
│  3 │ Ackley                      │ Multimodal   │ Any       │
│  4 │ Griewank                    │ Multimodal   │ Any       │
│  5 │ Zakharov                    │ Unimodal     │ Any       │
│  6 │ Schwefel_222                │ Unimodal     │ Any       │
│  7 │ Schwefel_12                 │ Unimodal     │ Any       │
│  8 │ Sum_Diff_Powers             │ Unimodal     │ Any       │
│  9 │ Matyas                      │ Unimodal     │ 2D        │
│ 10 │ Dixon_Price                 │ Unimodal     │ Any       │
│ 11 │ Levy                        │ Multimodal   │ Any       │
│ 12 │ Perm                        │ Multimodal   │ Any       │
│ 13 │ Rotated_Hyper_Ellipsoid     │ Unimodal     │ Any       │
│ 14 │ Bent_Cigar                  │ Unimodal     │ Any       │
│ 15 │ Booth                       │ Unimodal     │ 2D        │
└────┴─────────────────────────────┴──────────────┴───────────┘

Enter function number (or 0 to go back): 1
```

### Step 2: Running Analysis

```
╔════════════════════════════════════════════════════════════╗
║              ANALYZING: SPHERE FUNCTION                    ║
╚════════════════════════════════════════════════════════════╝

🔄 Executing Genetic Algorithm...
   • Population Size: 50
   • Generations: 1000
   • Dimensions: 5
   • Independent Runs: 30

Progress: [████████████████████████████████] 100%

Run  1/30 ✓ (seed=42)    Fitness: 0.123456
Run  2/30 ✓ (seed=84)    Fitness: 0.134567
Run  3/30 ✓ (seed=126)   Fitness: 0.145678
Run  4/30 ✓ (seed=168)   Fitness: 0.156789
Run  5/30 ✓ (seed=210)   Fitness: 0.167890
...
Run 28/30 ✓ (seed=1176)  Fitness: 0.309876
Run 29/30 ✓ (seed=1218)  Fitness: 0.320987
Run 30/30 ✓ (seed=1260)  Fitness: 0.332098

⏱️  Execution Time: 3.45 seconds
```

### Step 3: Results Table

```
╔════════════════════════════════════════════════════════════╗
║              STATISTICAL RESULTS - SPHERE                  ║
╚════════════════════════════════════════════════════════════╝

┌─────────────────────┬──────────────┬─────────────────────┐
│ Metric              │ Value        │ Interpretation      │
├─────────────────────┼──────────────┼─────────────────────┤
│ 🏆 Best (Min)       │ 0.123456     │ Excellent           │
│ 📊 Average (Mean)   │ 0.234567     │ Very Good           │
│ 📍 Median           │ 0.198765     │ Robust Center       │
│ 📈 Std Deviation    │ 0.045678     │ Highly Consistent   │
│ 🔢 Total Runs       │ 30           │ Statistical Valid   │
│ ⏱️  Time Taken      │ 3.45s        │ Fast                │
└─────────────────────┴──────────────┴─────────────────────┘

✅ All values normalized: 0 < value < 1 (excluding 0)
✅ Value range: [0.123456, 0.332098]
✅ Coefficient of Variation: 19.47% (Good consistency)

Performance Rating: ⭐⭐⭐⭐⭐ (Excellent)
```

### Step 4: Visualization

```
📊 Generating visualizations...

┌─────────────────────────────────────────────────────────────┐
│                    SPHERE FUNCTION ANALYSIS                 │
├──────────────────────────────┬──────────────────────────────┤
│  Convergence History         │  Distribution of Results     │
│                              │                              │
│  1.0 ┐                       │   Frequency                  │
│      │╲                      │    8 ┐                       │
│  0.8 │ ╲                     │      │     ┌──┐              │
│      │  ╲                    │    6 │   ┌─┤  ├─┐            │
│  0.6 │   ╲___                │      │   │ │  │ │            │
│      │       ╲___            │    4 │ ┌─┤ │  │ ├─┐          │
│  0.4 │           ╲___        │      │ │ │ │  │ │ │          │
│      │               ╲___    │    2 │ │ │ │  │ │ │          │
│  0.2 │                   ╲___│      │ │ │ │  │ │ │          │
│      │                      ╲│    0 └─┴─┴─┴──┴─┴─┴─         │
│  0.0 └───────────────────────┤      0.0  0.2  0.4  0.6      │
│      0   250  500  750  1000 │                              │
│      Generation              │  ── Mean: 0.234567           │
│                              │  ── Median: 0.198765         │
│  📦 Stats Box:               │                              │
│  Final: 0.123456             │  Histogram shows normal      │
│  Mean:  0.234567             │  distribution with slight    │
│  Std:   0.045678             │  right skew                  │
└──────────────────────────────┴──────────────────────────────┘

✓ Graphs displayed successfully!
```

---

## 📊 Demo 2: All Functions Combined

### Step 1: Initiate Combined Analysis

```
Enter your choice (1-3): 2

╔════════════════════════════════════════════════════════════╗
║         COMPREHENSIVE BENCHMARK ANALYSIS                   ║
╚════════════════════════════════════════════════════════════╝

🎯 Configuration:
   • Functions: 15
   • Runs per function: 30
   • Time Limit: 20 seconds
   • Total Evaluations: 450 runs

⚡ Optimizing execution strategy...
   Adjusted runs: 30 per function
   Estimated time: 18-20 seconds

🚀 Starting parallel execution...
```

### Step 2: Real-time Progress

```
════════════════════════════════════════════════════════════

Processing Functions:

[████████████████████] Sphere                    ✓ Done (1.2s)
[████████████████████] Rastrigin                 ✓ Done (1.3s)
[████████████████████] Ackley                    ✓ Done (1.2s)
[████████████████████] Griewank                  ✓ Done (1.4s)
[████████████████████] Zakharov                  ✓ Done (1.1s)
[████████████████████] Schwefel_222              ✓ Done (1.2s)
[████████████████████] Schwefel_12               ✓ Done (1.3s)
[████████████████████] Sum_Diff_Powers           ✓ Done (1.1s)
[████████████████████] Matyas                    ✓ Done (1.0s)
[████████████████████] Dixon_Price               ✓ Done (1.2s)
[████████████████████] Levy                      ✓ Done (1.3s)
[████████████████████] Perm                      ✓ Done (1.4s)
[████████████████████] Rotated_Hyper_Ellipsoid   ✓ Done (1.2s)
[████████████████████] Bent_Cigar                ✓ Done (1.1s)
[████████████████████] Booth                     ✓ Done (1.0s)

════════════════════════════════════════════════════════════

⏱️  Total Execution Time: 18.45 seconds
✅ All functions completed successfully!
```

### Step 3: Comprehensive Results Table

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                      COMPARATIVE PERFORMANCE ANALYSIS                                 ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

┌─────┬───────────────────────────┬──────────┬──────────┬──────────┬──────────┬────────┐
│ Rank│ Function                  │ Min      │ Mean     │ Median   │ Std Dev  │ Rating │
├─────┼───────────────────────────┼──────────┼──────────┼──────────┼──────────┼────────┤
│  1  │ 🥇 Sphere                 │ 0.001234 │ 0.012345 │ 0.010234 │ 0.005678 │ ⭐⭐⭐⭐⭐ │
│  2  │ 🥈 Sum_Diff_Powers        │ 0.002345 │ 0.015678 │ 0.013456 │ 0.006789 │ ⭐⭐⭐⭐⭐ │
│  3  │ 🥉 Schwefel_222           │ 0.003456 │ 0.018901 │ 0.016789 │ 0.007890 │ ⭐⭐⭐⭐⭐ │
│  4  │    Zakharov               │ 0.004567 │ 0.021234 │ 0.019012 │ 0.008901 │ ⭐⭐⭐⭐⭐ │
│  5  │    Schwefel_12            │ 0.005678 │ 0.024567 │ 0.022345 │ 0.009012 │ ⭐⭐⭐⭐⭐ │
│  6  │    Rotated_Hyper_Ellipsoid│ 0.006789 │ 0.027890 │ 0.025678 │ 0.010123 │ ⭐⭐⭐⭐  │
│  7  │    Bent_Cigar             │ 0.007890 │ 0.031234 │ 0.028901 │ 0.011234 │ ⭐⭐⭐⭐  │
│  8  │    Dixon_Price            │ 0.008901 │ 0.034567 │ 0.032234 │ 0.012345 │ ⭐⭐⭐⭐  │
│  9  │    Matyas                 │ 0.009012 │ 0.037890 │ 0.035567 │ 0.013456 │ ⭐⭐⭐⭐  │
│ 10  │    Booth                  │ 0.010123 │ 0.041234 │ 0.038890 │ 0.014567 │ ⭐⭐⭐⭐  │
│ 11  │    Griewank               │ 0.123456 │ 0.234567 │ 0.212345 │ 0.045678 │ ⭐⭐⭐   │
│ 12  │    Ackley                 │ 0.234567 │ 0.345678 │ 0.323456 │ 0.056789 │ ⭐⭐⭐   │
│ 13  │    Levy                   │ 0.345678 │ 0.456789 │ 0.434567 │ 0.067890 │ ⭐⭐    │
│ 14  │    Rastrigin              │ 0.456789 │ 0.567890 │ 0.545678 │ 0.078901 │ ⭐⭐    │
│ 15  │    Perm                   │ 0.567890 │ 0.678901 │ 0.656789 │ 0.089012 │ ⭐     │
└─────┴───────────────────────────┴──────────┴──────────┴──────────┴──────────┴────────┘

════════════════════════════════════════════════════════════════════════════════════════

📊 SUMMARY STATISTICS:

   🏆 Best Performing:     Sphere (Mean: 0.012345)
   📉 Most Challenging:    Perm (Mean: 0.678901)
   📊 Average Performance: 0.156789
   📈 Performance Range:   0.666667 (0.012345 to 0.678901)
   
   ✅ Unimodal Functions:  Generally better performance
   ⚠️  Multimodal Functions: More challenging (as expected)

════════════════════════════════════════════════════════════════════════════════════════
```

### Step 4: Combined Visualization

```
📊 Generating comprehensive visualization...

╔═══════════════════════════════════════════════════════════════════════════════════════╗
║              GA PERFORMANCE ANALYSIS - NORMALIZED VALUES (0-1)                        ║
╠═══════════════════════════════════════╦═══════════════════════════════════════════════╣
║  Best Fitness by Function             ║  Average Fitness by Function                  ║
║                                       ║                                               ║
║  Sphere            ▓░░░░░░░░░░        ║  Sphere            ▓▓░░░░░░░░░               ║
║  Sum_Diff_Powers   ▓░░░░░░░░░░        ║  Sum_Diff_Powers   ▓▓░░░░░░░░░               ║
║  Schwefel_222      ▓░░░░░░░░░░        ║  Schwefel_222      ▓▓░░░░░░░░░               ║
║  Zakharov          ▓░░░░░░░░░░        ║  Zakharov          ▓▓░░░░░░░░░               ║
║  Schwefel_12       ▓░░░░░░░░░░        ║  Schwefel_12       ▓▓░░░░░░░░░               ║
║  Rotated_HE        ▓░░░░░░░░░░        ║  Rotated_HE        ▓▓░░░░░░░░░               ║
║  Bent_Cigar        ▓░░░░░░░░░░        ║  Bent_Cigar        ▓▓▓░░░░░░░░               ║
║  Dixon_Price       ▓░░░░░░░░░░        ║  Dixon_Price       ▓▓▓░░░░░░░░               ║
║  Matyas            ▓░░░░░░░░░░        ║  Matyas            ▓▓▓░░░░░░░░               ║
║  Booth             ▓░░░░░░░░░░        ║  Booth             ▓▓▓▓░░░░░░░               ║
║  Griewank          ▓▓▓░░░░░░░░        ║  Griewank          ▓▓▓▓▓▓░░░░░               ║
║  Ackley            ▓▓▓▓░░░░░░░        ║  Ackley            ▓▓▓▓▓▓▓░░░░               ║
║  Levy              ▓▓▓▓▓░░░░░░        ║  Levy              ▓▓▓▓▓▓▓▓░░░               ║
║  Rastrigin         ▓▓▓▓▓▓░░░░░        ║  Rastrigin         ▓▓▓▓▓▓▓▓▓░░               ║
║  Perm              ▓▓▓▓▓▓▓░░░░        ║  Perm              ▓▓▓▓▓▓▓▓▓▓░               ║
║                                       ║                                               ║
║  0.0    0.2    0.4    0.6    0.8  1.0 ║  0.0    0.2    0.4    0.6    0.8    1.0      ║
╠═══════════════════════════════════════╬═══════════════════════════════════════════════╣
║  Convergence - Top 5 Functions        ║  Standard Deviation by Function               ║
║                                       ║                                               ║
║  1.0 ┐                                ║  Sphere            ▓░░░░░░░░░░               ║
║      │ ────── Sphere                 ║  Sum_Diff_Powers   ▓░░░░░░░░░░               ║
║  0.8 │ ────── Sum_Diff_Powers        ║  Schwefel_222      ▓░░░░░░░░░░               ║
║      │ ────── Schwefel_222           ║  Zakharov          ▓▓░░░░░░░░░               ║
║  0.6 │ ────── Zakharov               ║  Schwefel_12       ▓▓░░░░░░░░░               ║
║      │ ────── Schwefel_12            ║  Rotated_HE        ▓▓░░░░░░░░░               ║
║  0.4 │      ╲╲╲╲                     ║  Bent_Cigar        ▓▓░░░░░░░░░               ║
║      │         ╲╲╲╲___               ║  Dixon_Price       ▓▓▓░░░░░░░░               ║
║  0.2 │             ╲╲╲╲___           ║  Matyas            ▓▓▓░░░░░░░░               ║
║      │                 ╲╲╲╲___       ║  Booth             ▓▓▓░░░░░░░░               ║
║  0.0 └─────────────────────────      ║  Griewank          ▓▓▓▓▓░░░░░░               ║
║      0   250   500   750   1000      ║  Ackley            ▓▓▓▓▓▓░░░░░               ║
║      Generation                       ║  Levy              ▓▓▓▓▓▓▓░░░░               ║
║                                       ║  Rastrigin         ▓▓▓▓▓▓▓▓░░░               ║
║  All functions converge smoothly      ║  Perm              ▓▓▓▓▓▓▓▓▓░░               ║
║  Top performers reach low values      ║                                               ║
║  quickly and stabilize                ║  0.00   0.03   0.06   0.09                   ║
╚═══════════════════════════════════════╩═══════════════════════════════════════════════╝

✓ Visualization complete! All 4 graphs displayed.
```

### Step 5: Final Summary

```
╔════════════════════════════════════════════════════════════╗
║                  ANALYSIS COMPLETE                         ║
╚════════════════════════════════════════════════════════════╝

✅ Successfully analyzed 15 benchmark functions
✅ Performed 450 independent runs (30 per function)
✅ All values normalized to (0, 1) range, excluding 0
✅ Execution completed in 18.45 seconds (within 20s limit)

📊 Key Findings:
   • Unimodal functions: 10/15 (66.7%)
   • Multimodal functions: 5/15 (33.3%)
   • Average performance: 0.156789
   • Best performer: Sphere (0.012345)
   • Most challenging: Perm (0.678901)

🎯 Recommendations:
   • Sphere, Sum_Diff_Powers, Schwefel_222: Excellent for testing
   • Rastrigin, Levy, Perm: Good for challenging GA robustness
   • All functions show consistent convergence patterns

════════════════════════════════════════════════════════════

Press Enter to return to main menu...
```

---

## 🎨 Color Legend (Terminal Output)

```
🟢 Green  = Success / Excellent performance
🟡 Yellow = Warning / Moderate performance  
🔴 Red    = Error / Poor performance
🔵 Blue   = Information / Processing
⚪ White  = Normal text

Progress Bars:
[████████████████████] = Complete
[████████░░░░░░░░░░░░] = In Progress
[░░░░░░░░░░░░░░░░░░░░] = Pending
```

---

## 📈 Performance Interpretation

```
╔════════════════════════════════════════════════════════════╗
║              PERFORMANCE RATING GUIDE                      ║
╠════════════════════════════════════════════════════════════╣
║  Normalized Fitness    │  Rating  │  Interpretation       ║
╠════════════════════════╪══════════╪═══════════════════════╣
║  0.000 - 0.100         │  ⭐⭐⭐⭐⭐  │  Excellent            ║
║  0.100 - 0.300         │  ⭐⭐⭐⭐   │  Very Good            ║
║  0.300 - 0.500         │  ⭐⭐⭐    │  Good                 ║
║  0.500 - 0.700         │  ⭐⭐     │  Moderate             ║
║  0.700 - 1.000         │  ⭐      │  Challenging          ║
╚════════════════════════╧══════════╧═══════════════════════╝
```

---

## 🎯 Quick Tips

```
💡 TIP 1: Use Option 1 for detailed single-function analysis
💡 TIP 2: Use Option 2 for quick comparative overview
💡 TIP 3: Lower fitness values = Better performance
💡 TIP 4: Lower std dev = More consistent algorithm
💡 TIP 5: Check convergence graphs for optimization behavior
```

---

**End of Demo** 🎉

For actual execution, run: `python main_menu.py`
