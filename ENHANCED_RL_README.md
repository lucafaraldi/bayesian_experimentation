# Enhanced RL Agent for Constrained Bayesian Optimization

## Overview

This directory contains **enhanced training and benchmarking scripts** for RL-based constrained Bayesian optimization on COCO problems. The enhancements address key limitations identified in the original `train_rl_new.py` implementation.

## Files

- `train_rl_enhanced.py`: Enhanced training script with improved state representation and reward shaping
- `benchmark_rl_enhanced.py`: Benchmark script compatible with enhanced policies
- `benchmark_rl.py`: **Fixed** original benchmark (now correctly loads surrogate features and uses tanh)

## Key Enhancements

### 1. **Enhanced State Representation** (16 summary + 3 surrogate + 2*dim features)

**Original state (10 summary features):**
- `best_feasible`, `y_mean`, `y_std`, `y_max`
- `n_feasible`, `feasible_ratio`
- `n_obs`, `step`, `horizon`, `dim`

**Enhanced state (16 summary features) adds:**
- `c_mean`: Mean constraint violation (helps understand constraint landscape)
- `c_std`: Std of constraints (variability of violations)
- `c_min`: Minimum constraint value (how close to feasibility)
- `last_c`: Constraint at last evaluated point (immediate feedback)
- `best_c`: Constraint at best point (0.0 if feasible, else violation of least infeasible)
- `progress`: Fraction of budget used (encourages time-aware behavior)

**Why this helps:**
- Agent now "sees" constraint information, not just derived feasibility statistics
- Can learn to approach feasibility boundary systematically
- Understands trajectory of constraint violations

### 2. **Least Infeasible Point Tracking**

**Original:** When no feasible point exists, `best_point = zeros(dim)`

**Enhanced:** When no feasible point exists, `best_point = X[argmin(c)]`

**Why this helps:**
- Provides meaningful guidance toward feasibility
- Zeros are uninformative (especially problematic when domain is [0,1]^d)
- Agent learns which direction reduces constraint violation

### 3. **Improved Two-Stage Reward Function**

**Enhanced reward features:**

**Phase 1 (no feasible yet):**
- **First feasible bonus:** +10.0 (increased from +5.0)
- **Gradual violation penalty:** `-2.0 * tanh(c/10)` instead of binary
- **Approach bonus:** +0.5 if moving toward feasibility (c < min(prev_c))
- **Novelty bonus:** Encourages exploration

**Phase 2 (have feasible):**
- **Improvement reward:** `+5.0 * normalized_improvement`
- **Strong violation penalty:** `-3.0 * tanh(c/10)` (punish leaving feasible region)
- **Novelty bonus:** Continued exploration

**Why this helps:**
- Gradual penalties provide better gradient signal
- Approach bonus rewards progress even when still infeasible
- Stronger penalty for violating after finding feasibility prevents "wandering"

### 4. **Better Constraint Handling**

- Rewards scaled by `tanh(violation/10)` for smooth gradients
- Tracks constraint trajectory (improving vs worsening)
- Learns to stay near feasible region boundary

## Training Instructions

```bash
# Train enhanced policies for dim=2 and dim=10
python train_rl_enhanced.py
```

This will create:
- `models/coco_policy_enhanced_dim2.pt` (state_dim=23: 16+3+2+2)
- `models/coco_policy_enhanced_dim10.pt` (state_dim=39: 16+3+10+10)

Training config:
- Episodes: 10,000 per dimension
- Horizon: 15 RL steps per episode
- Initial design: 4*dim random points
- Optimizer: Adam, lr=3e-4
- Architecture: 2-layer MLP with hidden_dim=256

## Benchmarking Instructions

```bash
# Benchmark enhanced policies against qLogEI and Random
python benchmark_rl_enhanced.py
```

This runs:
- Functions: [2, 4, 6, 50, 52, 54]
- Instances: [1, 2, 3]
- Dimensions: [2, 10]
- Repetitions: 5
- Budget: 10*dim evaluations
- Initial random: 2*dim evaluations

Outputs:
- `results_coco_rl_enhanced_vs_qlogei.csv`: Per-evaluation results
- `results_coco_rl_enhanced_vs_qlogei_summary.csv`: Final values per repetition

## Critical Bugs Fixed in Original `benchmark_rl.py`

Even if you don't want to retrain, the original benchmark had critical bugs that are now fixed:

### Bug #1: Missing Surrogate Features
**Problem:** Training added 3 RBF regression features, but benchmark padded with zeros
**Impact:** Agent was "blind" to 3/17 features (17% of input!)
**Fix:** Implemented `_local_surrogate_features()` to match training

### Bug #2: Wrong Action Mapping
**Problem:** Training used `tanh` → [0,1], benchmark used `sigmoid` → [0,1]
**Impact:** Different output distributions, sigmoid has vanishing gradients
**Fix:** Changed to `tanh(action)` then rescale to [0,1]

### Bug #3: RNG Sharing
**Problem:** All algorithms shared same RNG, got different initial points
**Impact:** Unfair comparison
**Fix:** Each algorithm gets independent RNG with same seed

## State Dimension Comparison

| Version | Summary | Surrogate | Last | Best | Total |
|---------|---------|-----------|------|------|-------|
| Original | 10 | 3 | dim | dim | 13+2*dim |
| Enhanced | 16 | 3 | dim | dim | 19+2*dim |

**For dim=2:**
- Original: 10 + 3 + 2 + 2 = **17**
- Enhanced: 16 + 3 + 2 + 2 = **23**

**For dim=10:**
- Original: 10 + 3 + 10 + 10 = **33**
- Enhanced: 16 + 3 + 10 + 10 = **39**

## Expected Performance Improvements

The enhanced agent should:
1. ✓ **Find feasibility faster** (better constraint understanding)
2. ✓ **Stay feasible longer** (stronger violation penalties in Phase 2)
3. ✓ **Approach feasibility systematically** (least infeasible point guidance)
4. ✓ **Adapt to problem difficulty** (constraint statistics in state)
5. ✓ **Use budget wisely** (progress feature for time awareness)

## Compatibility Note

- `train_rl_enhanced.py` and `benchmark_rl_enhanced.py` are **paired** - use together
- `train_rl_new.py` and `benchmark_rl.py` (fixed version) are **paired** - use together
- Do NOT mix enhanced policies with original benchmark or vice versa (state_dim mismatch)

## Next Steps

1. **Train enhanced policies:** `python train_rl_enhanced.py` (~2-4 hours on CPU for both dims)
2. **Run benchmark:** `python benchmark_rl_enhanced.py`
3. **Compare results:** Check if enhanced agent beats qLogEI on more problems
4. **Analyze:** Look at feasibility rates and constraint satisfaction

## Summary of Changes

**Total new features in state:** +6 (c_mean, c_std, c_min, last_c, best_c, progress)

**State dim increase:**
- dim=2: 17 → 23 (+6 features)
- dim=10: 33 → 39 (+6 features)

**Reward improvements:**
- Gradual penalties (smooth gradients)
- Approach bonuses (reward progress)
- Stronger phase 2 penalties (maintain feasibility)

**Structural improvements:**
- Least infeasible tracking (better guidance)
- Constraint statistics (landscape understanding)
- Time awareness (progress metric)
