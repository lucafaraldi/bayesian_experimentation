# Guide: Comparing RL Versions

## Overview

You have three RL versions to compare:

1. **Original RL** (`benchmark_rl.py`)
   - Uses `models/coco_policy_dim{2,10}.pt`
   - Basic state representation (10 summary + 3 surrogate + 2*dim)
   - Trained on unknown data (legacy)

2. **Enhanced RL** (`benchmark_rl_enhanced.py`)
   - Uses `models/coco_policy_enhanced_dim{2,10}.pt`
   - Enhanced state (16 summary + 3 surrogate + 2*dim = 23 for d=2)
   - Trained on 6 COCO functions, 3 instances, 10k episodes

3. **Full COCO RL** (`benchmark_rl_full_coco.py`)
   - Uses `models/coco_policy_full_dim{2,10}.pt`
   - Same enhanced state representation
   - Trained on ALL 54 COCO functions, 5 instances, 20k episodes

## Fair Comparison Setup

All three benchmarks use **identical test conditions**:

```python
FUNCTIONS = [2, 4, 6, 50, 52, 54]  # Same test functions
INSTANCES = [1, 2, 3]              # Same instances
DIMENSIONS = [2, 10]               # Same dimensions
REPETITIONS = 5                    # Same number of runs
BUDGET_FACTOR = 10                 # Same budget
N_INIT_FACTOR = 2                  # Same initial design
seed = 1234 + rep                  # Same random seeds
```

**This guarantees a fair comparison!**

## Step-by-Step Comparison

### Step 1: Run All Three Benchmarks

```bash
cd /Users/lucafaraldi/Documents/projects/leiden/constrained_bo_0/bayesian_experimentation

# Original RL
python benchmark_rl.py
# → results_coco_rl_vs_qlogei_summary.csv

# Enhanced RL (6 functions training)
python benchmark_rl_enhanced.py
# → results_coco_rl_enhanced_vs_qlogei_summary.csv

# Full COCO RL (all functions training)
python benchmark_rl_full_coco.py
# → results_coco_rl_full_vs_qlogei_summary.csv
```

**Time estimate**: ~30-60 minutes per benchmark

### Step 2: Run Comparison Analysis

```bash
python compare_rl_versions.py
```

This will output:
- **Overall statistics**: Mean, median, std dev per version
- **Win rates**: How often each version beats the others
- **Per-function analysis**: Which version is best for each function
- **Per-dimension analysis**: Performance by dimension
- **Statistical tests**: Paired t-tests with p-values and effect sizes

### Step 3: Interpret Results

#### Example Output:

```
======================================================================
OVERALL STATISTICS
======================================================================
    Version         Mean      Median         Std  N_finite  N_inf
   Original  -1.234e+03  -8.765e+02  5.432e+02        75      15
   Enhanced  -9.876e+02  -6.543e+02  4.321e+02        78      12
  Full_COCO  -8.765e+02  -5.432e+02  3.210e+02        82       8

======================================================================
WIN RATE ANALYSIS
======================================================================
Total problems: 90

Win counts (lower best_feasible wins):
  Full_COCO      : 45 / 90 (50.0%)
  Enhanced       : 30 / 90 (33.3%)
  Original       : 15 / 90 (16.7%)
```

**Interpretation:**
- Lower mean/median = better performance
- Higher win rate = more often beats other methods
- Fewer inf values = finds feasible solutions more often

#### Statistical Significance:

```
Enhanced vs Full_COCO:
  Mean difference: 1.111e+02 (positive = Full_COCO better)
  p-value: 0.0234
  Cohen's d: 0.342
  ✓ Statistically significant (p < 0.05)
  Winner: Full_COCO
```

**Interpretation:**
- p < 0.05: Difference is statistically significant (not due to chance)
- Cohen's d > 0.3: Medium effect size (meaningful difference)
- Positive mean difference: Full_COCO has better (lower) scores

## What to Expect

### Expected Results:

1. **Full COCO ≥ Enhanced > Original**
   - Full COCO should generalize best (trained on most data)
   - Enhanced should beat Original (better state representation)

2. **Full COCO Strengths:**
   - Better on functions NOT in Enhanced training set
   - More robust across different problem types
   - Fewer inf values (finds feasibility more often)

3. **Full COCO Weaknesses:**
   - May be slightly worse on specific functions in Enhanced training
   - Training on 54 functions → less overfitting to 6 test functions

### Surprising Results:

**If Enhanced beats Full COCO:**
- This suggests the test set (6 functions) is too similar to Enhanced training set
- Full COCO trades off specialization for generalization
- Solution: Test on MORE functions (edit `FUNCTIONS` in benchmarks)

**If Original beats both:**
- Check if Original model is actually different from Enhanced
- May indicate a bug in Enhanced/Full training

## Advanced Analysis

### Test Generalization on Unseen Functions

Edit benchmark configs to test on functions NOT in Enhanced training:

```python
# Enhanced was trained on: [2, 4, 6, 50, 52, 54]
# Test on completely new functions:
FUNCTIONS = [8, 10, 12, 14, 16, 18]  # Unseen by Enhanced
```

Run benchmarks again. Full COCO should dominate here!

### Visualize Learning Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load full results (per-eval, not summary)
df_enh = pd.read_csv('results_coco_rl_enhanced_vs_qlogei.csv')
df_full = pd.read_csv('results_coco_rl_full_vs_qlogei.csv')

# Plot learning curves for a specific problem
problem_df_enh = df_enh[(df_enh['function'] == 2) &
                         (df_enh['instance'] == 1) &
                         (df_enh['dim'] == 2) &
                         (df_enh['repetition'] == 0)]

problem_df_full = df_full[(df_full['function'] == 2) &
                          (df_full['instance'] == 1) &
                          (df_full['dim'] == 2) &
                          (df_full['repetition'] == 0)]

# Plot
for method in ['RL_Enhanced', 'RL_Full', 'qLogEI', 'Random']:
    if method in problem_df_enh['method'].values:
        subset = problem_df_enh[problem_df_enh['method'] == method]
        plt.plot(subset['eval'], subset['best_feasible'], label=method)
    elif method in problem_df_full['method'].values:
        subset = problem_df_full[problem_df_full['method'] == method]
        plt.plot(subset['eval'], subset['best_feasible'], label=method)

plt.xlabel('Evaluation')
plt.ylabel('Best Feasible (lower is better)')
plt.legend()
plt.title('F2 instance 1, dim=2')
plt.yscale('symlog')  # Handle both positive and negative values
plt.grid(True, alpha=0.3)
plt.savefig('learning_curves.png')
plt.show()
```

### Export Results for Paper/Report

```python
import pandas as pd

# Load all results
df_orig = pd.read_csv('results_coco_rl_vs_qlogei_summary.csv')
df_enh = pd.read_csv('results_coco_rl_enhanced_vs_qlogei_summary.csv')
df_full = pd.read_csv('results_coco_rl_full_vs_qlogei_summary.csv')

# Extract RL only
orig_rl = df_orig[df_orig['method'] == 'RL'][['function', 'instance', 'dim', 'repetition', 'final_best_feasible']]
orig_rl.columns = ['function', 'instance', 'dim', 'repetition', 'Original_RL']

enh_rl = df_enh[df_enh['method'] == 'RL_Enhanced'][['function', 'instance', 'dim', 'repetition', 'final_best_feasible']]
enh_rl.columns = ['function', 'instance', 'dim', 'repetition', 'Enhanced_RL']

full_rl = df_full[df_full['method'] == 'RL_Full'][['function', 'instance', 'dim', 'repetition', 'final_best_feasible']]
full_rl.columns = ['function', 'instance', 'dim', 'repetition', 'Full_COCO_RL']

# Merge
merged = orig_rl.merge(enh_rl, on=['function', 'instance', 'dim', 'repetition'], how='outer')
merged = merged.merge(full_rl, on=['function', 'instance', 'dim', 'repetition'], how='outer')

# Add qLogEI baseline
qlogei = df_full[df_full['method'] == 'qLogEI'][['function', 'instance', 'dim', 'repetition', 'final_best_feasible']]
qlogei.columns = ['function', 'instance', 'dim', 'repetition', 'qLogEI']
merged = merged.merge(qlogei, on=['function', 'instance', 'dim', 'repetition'], how='left')

# Compute aggregate stats
aggregated = merged.groupby(['function', 'instance', 'dim']).agg({
    'Original_RL': ['mean', 'std'],
    'Enhanced_RL': ['mean', 'std'],
    'Full_COCO_RL': ['mean', 'std'],
    'qLogEI': ['mean', 'std'],
})

# Save for LaTeX table
aggregated.to_csv('results_for_paper.csv')
print(aggregated.to_latex(float_format="%.2e"))
```

## Summary: How to Ensure Fair Comparison

✅ **Already done automatically:**
1. Same test functions, instances, dimensions
2. Same random seeds (`seed = 1234 + rep`)
3. Same budget and initial design size
4. Same evaluation procedure

✅ **What you need to do:**
1. Run all three benchmarks
2. Use `compare_rl_versions.py` to analyze
3. Check statistical significance
4. Interpret win rates and effect sizes

✅ **Key metrics to report:**
- Mean/median final best feasible (lower is better)
- Win rate (% of problems where method wins)
- Number of inf values (fewer is better)
- Statistical significance (p-value < 0.05)
- Effect size (Cohen's d > 0.3 is meaningful)

## Troubleshooting

**Q: Results files have different numbers of rows**
A: This is OK if some methods are missing in certain files. The comparison script handles this.

**Q: Getting many inf values**
A: This is normal for hard COCO problems. Focus on win rates among finite values.

**Q: Full COCO doesn't beat Enhanced**
A: Test on more functions outside the Enhanced training set. Edit `FUNCTIONS = [8, 10, 12, ...]`

**Q: Statistical tests say "not significant"**
A: Either difference is real but small, or sample size (5 reps) is too small. Try REPETITIONS = 10.

## Quick Start

After training completes:

```bash
# 1. Run benchmarks (90 minutes total)
python benchmark_rl.py
python benchmark_rl_enhanced.py
python benchmark_rl_full_coco.py

# 2. Compare results (instant)
python compare_rl_versions.py

# 3. Done! Check the output.
```

That's it! You now have a rigorous, statistically sound comparison of all three RL versions. 🎉
