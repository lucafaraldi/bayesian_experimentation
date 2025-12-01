# Quick Start: Full COCO RL Training

## What's Different from Enhanced?

**Enhanced version** (`train_rl_enhanced.py`):
- Trains on 6 COCO functions only
- Uses instances 1-3
- 10,000 episodes
- Fast training (~2-3 hours)
- May overfit to specific functions

**Full COCO version** (`train_rl_full_coco.py`):
- Trains on ALL available COCO functions (~54 functions)
- Uses instances 1-5
- 20,000 episodes
- Longer training (~8-12 hours)
- Better generalization to unseen problems

## Step 1: Train Full COCO Policies

```bash
cd /Users/lucafaraldi/Documents/projects/leiden/constrained_bo_0/bayesian_experimentation
python train_rl_full_coco.py
```

**What this does:**
- Samples from entire COCO bbob-constrained suite
- Trains policies for dim=2 and dim=10
- 20,000 episodes per dimension (2x enhanced version)
- Takes 8-12 hours on CPU (3-5 hours with GPU)
- Saves to `models/coco_policy_full_dim{2,10}.pt`

**You'll see output like:**
```
======================================================================
Training Full COCO Policy for dim=2
======================================================================
[INFO] Found 54 problems for dim=2

Episode 500/20000, Avg Reward (last 500): -3.45
Episode 1000/20000, Avg Reward (last 500): -1.23
Episode 5000/20000, Avg Reward (last 500): 1.87
...
Saved full COCO policy to models/coco_policy_full_dim2.pt
State dim: 23
```

**Note:** You'll see many different function IDs being sampled (F2, F4, F6, F8, ... F54) unlike the enhanced version which only uses F2, F4, F6, F50, F52, F54.

## Step 2: Run Benchmark

```bash
python benchmark_rl_full_coco.py
```

**What this does:**
- Tests on same 6 functions × 3 instances × 2 dims × 5 reps = 180 problems
- Compares RL_Full vs qLogEI vs Random
- Takes ~30-60 minutes
- Saves results to CSV files

**Output:**
```
=== DIM = 2, budget = 20, n_init = 4 ===
[INFO] Loaded FULL COCO RL model: models/coco_policy_full_dim2.pt

Problem F2, instance 1, dim 2
  Repetition 1/5...
    Summary: Random=-1.814e+03, qLogEI=-2.009e+03, RL_Full=-2.156e+03
```

## Expected Performance Gains

### Full COCO Should Be Better At:
1. **Generalization**: Better on test functions not seen during training
2. **Robustness**: Less sensitive to specific problem characteristics
3. **Diverse constraints**: Better at handling different constraint patterns
4. **Rare problem types**: Better on unusual COCO functions

### Trade-offs:
1. **Training time**: 2-4x longer than enhanced version
2. **Memory**: Needs to store more training examples
3. **Convergence**: May be slower to converge on specific functions
4. **Overfitting**: Less likely to overfit, but may underfit individual problems

## Comparison: Enhanced vs Full COCO

| Feature | Enhanced | Full COCO |
|---------|----------|-----------|
| Training functions | 6 | ~54 |
| Training instances | 1-3 | 1-5 |
| Episodes | 10,000 | 20,000 |
| Training time | 2-3 hours | 8-12 hours |
| State representation | Same (23 for d=2) | Same (23 for d=2) |
| Reward function | Same | Same |
| Generalization | Good | Better |

## Analyzing Results

```python
import pandas as pd
import numpy as np

# Load results
df_enhanced = pd.read_csv('results_coco_rl_enhanced_vs_qlogei_summary.csv')
df_full = pd.read_csv('results_coco_rl_full_vs_qlogei_summary.csv')

# Compare Enhanced vs Full COCO
enhanced_rl = df_enhanced[df_enhanced['method'] == 'RL_Enhanced']['final_best_feasible']
full_rl = df_full[df_full['method'] == 'RL_Full']['final_best_feasible']

print("Enhanced RL - Mean:", enhanced_rl.mean())
print("Full COCO RL - Mean:", full_rl.mean())

# Win rates
pivot_enhanced = df_enhanced.pivot_table(
    values='final_best_feasible',
    index=['function', 'instance', 'repetition'],
    columns='method'
)
pivot_full = df_full.pivot_table(
    values='final_best_feasible',
    index=['function', 'instance', 'repetition'],
    columns='method'
)

# Which is better?
if 'RL_Enhanced' in pivot_enhanced.columns:
    enhanced_vs_qlogei = (pivot_enhanced['RL_Enhanced'] < pivot_enhanced['qLogEI']).mean()
    print(f"\nEnhanced RL wins over qLogEI: {enhanced_vs_qlogei:.1%}")

if 'RL_Full' in pivot_full.columns:
    full_vs_qlogei = (pivot_full['RL_Full'] < pivot_full['qLogEI']).mean()
    print(f"Full COCO RL wins over qLogEI: {full_vs_qlogei:.1%}")
```

## Tips for Better Results

1. **Even longer training:** Increase `N_EPISODES = 30_000` for better convergence
2. **More functions in benchmark:** Edit `FUNCTIONS` in `benchmark_rl_full_coco.py` to test on functions not seen during training
3. **Hyperparameter tuning:**
   - `hidden_dim=512` for more capacity (currently 256)
   - `lr=5e-5` for more stable learning (currently 1e-4)
   - `clip_eps=0.15` for less conservative updates (currently 0.2)

## Debugging

**If training takes too long:**
```bash
# Check how many problems are being used
python -c "
import cocoex
suite = cocoex.Suite('bbob-constrained', '', '')
problems_d2 = [p for p in suite if p.dimension == 2]
print(f'Found {len(problems_d2)} problems for dim=2')
"
```

**If getting worse results than enhanced:**
- This is normal in early training (first 5000 episodes)
- Full COCO explores more diverse problems
- May need longer training to converge
- Check reward curves - should eventually exceed enhanced

**If models not found:**
```bash
ls -la models/coco_policy_full_dim*.pt
# Should show both dim2 and dim10 .pt files
```

## Files Generated

After training and benchmarking:
```
models/
  coco_policy_full_dim2.pt     # Trained on ALL COCO for d=2
  coco_policy_full_dim10.pt    # Trained on ALL COCO for d=10

results_coco_rl_full_vs_qlogei.csv          # Per-eval results
results_coco_rl_full_vs_qlogei_summary.csv  # Final results per run
```

## Next Steps

1. **Compare all three versions:**
   ```python
   import pandas as pd

   df_orig = pd.read_csv('results_coco_rl_vs_qlogei_summary.csv')
   df_enh = pd.read_csv('results_coco_rl_enhanced_vs_qlogei_summary.csv')
   df_full = pd.read_csv('results_coco_rl_full_vs_qlogei_summary.csv')

   # Extract RL results
   orig_rl = df_orig[df_orig['method'] == 'RL']['final_best_feasible'].mean()
   enh_rl = df_enh[df_enh['method'] == 'RL_Enhanced']['final_best_feasible'].mean()
   full_rl = df_full[df_full['method'] == 'RL_Full']['final_best_feasible'].mean()

   print(f"Original RL: {orig_rl:.3e}")
   print(f"Enhanced RL: {enh_rl:.3e}")
   print(f"Full COCO RL: {full_rl:.3e}")
   ```

2. **Test on completely new functions:**
   - Edit `FUNCTIONS` in benchmark to use F8, F10, etc.
   - These functions were not in the enhanced training set
   - Full COCO should generalize better

3. **Visualize training progress:**
   - During training, episode rewards are printed
   - Plot these to see convergence
   - Compare convergence speed of enhanced vs full

## Summary

You now have three versions:
- ✅ **Original (Fixed)**: `train_rl_new.py` - basic implementation with bug fixes
- ✅ **Enhanced**: `train_rl_enhanced.py` - better state/rewards, 6 functions
- ✅ **Full COCO**: `train_rl_full_coco.py` - best generalization, all functions

**Which to use:**
- **Enhanced**: Fast training, good for testing ideas
- **Full COCO**: Best generalization, production use

Ready to train on the full COCO suite? Just run:
```bash
python train_rl_full_coco.py
```

Expected training time: 8-12 hours (CPU) or 3-5 hours (GPU) 🚀
