# Answer: How to Guarantee Fair Comparison

## Your Question

> "How can I guarantee that after I train the new agent on the whole COCO set, I can fairly compare the performance on our benchmark between the former ones (trained only on specific functions) and the new ones?"

## Short Answer

**It's already guaranteed!** ✅

All three benchmark scripts (`benchmark_rl.py`, `benchmark_rl_enhanced.py`, `benchmark_rl_full_coco.py`) use **identical test configurations**:

```python
# Same for all three benchmarks:
FUNCTIONS = [2, 4, 6, 50, 52, 54]  # Same test problems
INSTANCES = [1, 2, 3]              # Same instances
DIMENSIONS = [2, 10]               # Same dimensions
REPETITIONS = 5                    # Same number of runs
BUDGET_FACTOR = 10                 # Same evaluation budget
N_INIT_FACTOR = 2                  # Same initial random design
seed = 1234 + rep                  # Same random seeds per repetition
```

**This means:**
- All versions are tested on the **exact same problems**
- All versions use the **exact same random seeds**
- All versions have the **exact same computational budget**
- All versions start with the **exact same initial random points**

## What You Need to Do

### Step 1: Run All Three Benchmarks

After your Full COCO training finishes:

```bash
cd /Users/lucafaraldi/Documents/projects/leiden/constrained_bo_0/bayesian_experimentation

# Test Original RL
python benchmark_rl.py
# → Output: results_coco_rl_vs_qlogei_summary.csv

# Test Enhanced RL (trained on 6 functions)
python benchmark_rl_enhanced.py
# → Output: results_coco_rl_enhanced_vs_qlogei_summary.csv

# Test Full COCO RL (trained on 54 functions)
python benchmark_rl_full_coco.py
# → Output: results_coco_rl_full_vs_qlogei_summary.csv
```

**Time**: ~30 minutes per benchmark = 90 minutes total

### Step 2: Compare Results

```bash
python compare_rl_versions.py
```

This automatically:
- ✅ Computes statistics (mean, median, std dev)
- ✅ Calculates win rates (which version wins most often)
- ✅ Performs statistical tests (paired t-tests, p-values, effect sizes)
- ✅ Analyzes per-function and per-dimension performance

## Why This Guarantees Fair Comparison

### 1. **Same Random Seeds**
```python
seed = 1234 + rep  # rep = 0, 1, 2, 3, 4
```

For repetition 0, all three versions use seed 1234:
- Same initial random points from n_init design
- Same RNG state for all algorithms (Random, qLogEI, RL)
- Exact reproducibility

### 2. **Same Test Problems**
All three benchmarks test on:
- 6 functions × 3 instances × 2 dimensions × 5 repetitions = **180 runs**

This is the **only** difference between training sets:
- **Enhanced**: Trained on functions [2, 4, 6, 50, 52, 54]
- **Full COCO**: Trained on ALL 54 functions

But **testing is identical!**

### 3. **Same Computational Budget**
- Budget = 10 × dim (20 for d=2, 100 for d=10)
- Initial design = 2 × dim (4 for d=2, 20 for d=10)
- Sequential evaluations = Budget - n_init

All methods get exactly the same number of function evaluations.

### 4. **Same Evaluation Procedure**
All three use the same code in `run_single_algorithm()`:
1. Start with n_init random points (same seed)
2. For each iteration:
   - Ask optimizer for next point
   - Evaluate on COCO problem
   - Track best feasible solution
3. Return best feasible trajectory

## What the Comparison Script Does

`compare_rl_versions.py` provides:

### 1. Overall Statistics
```
Version         Mean      Median  N_finite  N_inf
Original  -1.234e+03  -8.765e+02       75     15
Enhanced  -9.876e+02  -6.543e+02       78     12
Full_COCO -8.765e+02  -5.432e+02       82      8
```

**Lower is better!**

### 2. Win Rates
```
Win counts (lower best_feasible wins):
  Full_COCO : 45 / 90 (50.0%)  ← Wins half the problems
  Enhanced  : 30 / 90 (33.3%)
  Original  : 15 / 90 (16.7%)
```

### 3. Statistical Significance
```
Enhanced vs Full_COCO:
  p-value: 0.0234
  Cohen's d: 0.342
  ✓ Statistically significant (p < 0.05)
  Winner: Full_COCO
```

**p < 0.05** means the difference is **NOT due to chance**!

### 4. Per-Function Breakdown
Shows which version is best for each COCO function.

## Expected Results

### Hypothesis:
**Full COCO ≥ Enhanced > Original**

### Why Full COCO Should Win:

1. **More training data**: 270 problems vs 18 problems
2. **Better generalization**: Sees diverse constraint patterns
3. **Less overfitting**: Can't memorize 270 problems

### Why Full COCO Might Lose on Specific Functions:

- Enhanced can **overfit** to the 6 test functions
- Enhanced trains for 10k episodes **only on test functions**
- Full COCO trains for 20k episodes **across all functions**

**This is the generalization vs specialization trade-off!**

## Testing True Generalization

Want to test which version **truly generalizes**?

Edit the benchmark to test on **unseen functions**:

```python
# In benchmark_rl_full_coco.py and benchmark_rl_enhanced.py:
FUNCTIONS = [8, 10, 12, 14, 16, 18]  # NOT in Enhanced training!
```

Now:
- **Enhanced**: Never saw these functions during training
- **Full COCO**: Trained on these functions

**Full COCO should dominate here!**

## Interpreting Results

### If Full COCO Wins:
✅ Training on entire COCO suite improves generalization
✅ 20k episodes + 270 problems > 10k episodes + 18 problems
✅ Worth the extra training time

### If Enhanced Wins:
⚠️ Test set (6 functions) is too similar to Enhanced training
⚠️ Full COCO trades specialization for generalization
→ Solution: Test on more diverse functions

### If Original Wins:
🚨 Something is wrong - check that models are different
🚨 May indicate bug in Enhanced/Full training

## Key Takeaways

### What Guarantees Fairness:
1. ✅ Same test problems
2. ✅ Same random seeds
3. ✅ Same budget
4. ✅ Same evaluation procedure
5. ✅ Statistical testing (p-values, effect sizes)

### What You Need to Do:
1. Run three benchmark scripts (90 minutes)
2. Run comparison script (instant)
3. Interpret results

### What to Report:
- Mean/median final best feasible (lower is better)
- Win rate (% of problems won)
- Statistical significance (p-value)
- Effect size (Cohen's d)

## Files Overview

### Training Scripts:
- `train_rl_new.py` → `models/coco_policy_dim{2,10}.pt`
- `train_rl_enhanced.py` → `models/coco_policy_enhanced_dim{2,10}.pt`
- `train_rl_full_coco.py` → `models/coco_policy_full_dim{2,10}.pt`

### Benchmark Scripts:
- `benchmark_rl.py` → Uses Original models
- `benchmark_rl_enhanced.py` → Uses Enhanced models
- `benchmark_rl_full_coco.py` → Uses Full COCO models

### Analysis Scripts:
- `compare_rl_versions.py` → Statistical comparison
- `COMPARISON_GUIDE.md` → This guide

## Quick Start After Training

```bash
# Run benchmarks (can run in parallel if you have multiple cores)
python benchmark_rl.py &
python benchmark_rl_enhanced.py &
python benchmark_rl_full_coco.py &
wait

# Compare results
python compare_rl_versions.py

# Done! Check the output for:
# - Win rates
# - Statistical significance
# - Per-function performance
```

## Bottom Line

**Your comparison is already fair!** All three benchmarks use:
- ✅ Identical test problems
- ✅ Identical random seeds
- ✅ Identical budgets
- ✅ Identical evaluation procedures

The **only** difference is which checkpoint file is loaded:
- Original: `coco_policy_dim{dim}.pt`
- Enhanced: `coco_policy_enhanced_dim{dim}.pt`
- Full COCO: `coco_policy_full_dim{dim}.pt`

This is exactly what you want for a fair comparison! 🎉
