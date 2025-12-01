# Quick Start: Enhanced RL for Constrained BO

## What We Built

**Two versions of RL agent:**
1. **Original (Fixed)**: `train_rl_new.py` + `benchmark_rl.py` (bugs fixed)
2. **Enhanced**: `train_rl_enhanced.py` + `benchmark_rl_enhanced.py` (new features)

## Step 1: Train Enhanced Policies

```bash
cd /Users/lucafaraldi/Documents/projects/leiden/constrained_bo_0/bayesian_experimentation
python train_rl_enhanced.py
```

**What this does:**
- Trains policies for dim=2 and dim=10
- ~10,000 episodes per dimension
- Takes 2-4 hours on CPU (faster with GPU)
- Saves to `models/coco_policy_enhanced_dim{2,10}.pt`

**You'll see output like:**
```
======================================================================
Training Enhanced Policy for dim=2
======================================================================

Episode 500/10000, Avg Reward (last 500): 2.34
Episode 1000/10000, Avg Reward (last 500): 3.12
...
Saved enhanced policy to models/coco_policy_enhanced_dim2.pt
State dim: 23
```

## Step 2: Run Benchmark

```bash
python benchmark_rl_enhanced.py
```

**What this does:**
- Tests on 6 functions × 3 instances × 2 dims × 5 reps = 180 problem instances
- Compares RL_Enhanced vs qLogEI vs Random
- Takes ~30-60 minutes
- Saves results to CSV files

**Output:**
```
=== DIM = 2, budget = 20, n_init = 4 ===
[INFO] Loaded ENHANCED RL model: models/coco_policy_enhanced_dim2.pt

Problem F2, instance 1, dim 2
  Repetition 1/5...
    Summary: Random=-1.814e+03, qLogEI=-2.009e+03, RL_Enhanced=-2.156e+03
```

## Step 3: Analyze Results

```python
import pandas as pd

# Load summary
df = pd.read_csv('results_coco_rl_enhanced_vs_qlogei_summary.csv')

# Compare methods
comparison = df.groupby('method')['final_best_feasible'].agg(['mean', 'std', 'count'])
print(comparison)

# Check feasibility rates
full_df = pd.read_csv('results_coco_rl_enhanced_vs_qlogei.csv')
# Filter to final evaluations
final = full_df[full_df['eval'] == full_df['eval'].max()]
# Count non-inf values (feasible solutions found)
feas_rate = final.groupby('method')['best_feasible'].apply(lambda x: (~np.isinf(x)).mean())
print("\nFeasibility rates:")
print(feas_rate)
```

## What to Expect

### Enhanced RL Should Be Better At:
1. **Finding feasibility faster** (constraint statistics help)
2. **Staying feasible** (stronger penalties in Phase 2)
3. **Hard problems** (least infeasible guidance when struggling)
4. **Budget efficiency** (progress awareness)

### qLogEI Might Still Win On:
1. **Objective optimization** (GP uncertainty quantification)
2. **Problems where feasibility is easy** (RL overhead not worth it)
3. **Exploitation** (RL still explores more)

## Comparison: Original vs Enhanced

| Feature | Original | Enhanced |
|---------|----------|----------|
| State dim (d=2) | 17 | 23 |
| Constraint features | 0 | 6 |
| Infeas. guidance | Zeros | Least violated |
| Reward | Binary penalties | Gradual + approach bonus |
| Phase 2 penalty | -0.5 * tanh(c) | -3.0 * tanh(c) |

## Debugging

**If training fails:**
```bash
# Check COCO installation
python -c "import cocoex; print(cocoex.__version__)"

# Check PyTorch
python -c "import torch; print(torch.__version__)"
```

**If benchmark can't find models:**
```bash
ls -la models/coco_policy_enhanced_dim*.pt
# Should show both dim2 and dim10 .pt files
```

**If getting dimension mismatch errors:**
- Make sure you're using `benchmark_rl_enhanced.py` with enhanced policies
- Original policies (17/33 dims) won't work with enhanced benchmark (23/39 dims)

## Tips for Better Results

1. **Longer training:** Increase `N_EPISODES = 20_000` for better policies
2. **Different seeds:** Change `GLOBAL_SEED` to explore different initializations
3. **Hyperparameter tuning:**
   - Try `hidden_dim=512` for more capacity
   - Adjust `lr=1e-4` for more stable learning
   - Tune `clip_eps=0.1` for more conservative updates

## Next Steps

1. **Visualize learning curves:**
   ```python
   import matplotlib.pyplot as plt
   # Plot episode_rewards from training
   ```

2. **Per-problem analysis:**
   ```python
   # Which problems does RL beat qLogEI on?
   df = pd.read_csv('results_coco_rl_enhanced_vs_qlogei_summary.csv')
   pivot = df.pivot_table(
       values='final_best_feasible',
       index=['function', 'instance'],
       columns='method'
   )
   pivot['RL_wins'] = pivot['RL_Enhanced'] < pivot['qLogEI']
   print(pivot)
   ```

3. **Train on more problems:**
   - Edit `FUNCTIONS` in `train_rl_enhanced.py`
   - Add more COCO functions to training distribution

## Files Generated

After training and benchmarking:
```
models/
  coco_policy_enhanced_dim2.pt     # Trained policy for d=2
  coco_policy_enhanced_dim10.pt    # Trained policy for d=10

results_coco_rl_enhanced_vs_qlogei.csv          # Per-eval results
results_coco_rl_enhanced_vs_qlogei_summary.csv  # Final results per run
```

## Summary

You now have:
- ✅ **Fixed original implementation** (bugs corrected in `benchmark_rl.py`)
- ✅ **Enhanced implementation** (better state + rewards in `train_rl_enhanced.py`)
- ✅ **Benchmark infrastructure** (fair comparison with qLogEI and Random)
- ✅ **Documentation** (this guide + `ENHANCED_RL_README.md`)

Ready to train? Just run:
```bash
python train_rl_enhanced.py
```

Good luck! 🚀
