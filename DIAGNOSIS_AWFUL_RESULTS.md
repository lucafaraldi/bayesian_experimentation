# Diagnosis: Why Your RL Results Are Awful

## Summary of Results

Your RL policy **performs worse than both baselines** (qLogEI and Random Search):

| Metric | Random | qLogEI | RL | Winner |
|--------|--------|--------|-----|--------|
| **Feasibility Rate** | 48.9% | **53.9%** | 35.0% | qLogEI ✓ |
| **Avg Performance (feasible)** | -1308.6 | **-1120.0** | -2113.5 | qLogEI ✓ |
| **D=2 Feasibility** | 66.7% | **77.8%** | 46.7% | qLogEI ✓ |
| **D=10 Feasibility** | 31.1% | **30.0%** | 23.3% | Random ✓ |

**Result:** RL is WORSE than both baselines on all metrics.

---

## Why Is This Happening?

### Problem 1: RL Fails to Find Feasible Solutions (117/180 runs = 65% failure rate!)

**Likely Causes:**

1. **Reward Engineering Issues:**
   - Your reward gives +5.0 for first feasible point, but this might not be strong enough
   - The penalty for infeasible points (-1.0 × violation) may be too weak
   - RL learned to explore randomly without properly exploiting the GP information

2. **Training Distribution Mismatch:**
   - RL was trained on 10k episodes with horizon=15 steps
   - But at test time, budget is only 10×D (20 evals for D=2, 100 for D=10)
   - The policy might not have learned to find feasibility quickly enough

3. **State Representation Problems:**
   - The 13 + 2D state features might not capture enough information
   - Local RBF surrogate with only 5 centers is very crude
   - Missing critical GP uncertainty information

### Problem 2: When RL Finds Feasible Solutions, They're Low Quality

**Likely Causes:**

1. **Exploration vs Exploitation Imbalance:**
   - Novelty bonus (+0.1 to +0.2) encourages exploration
   - But this might prevent exploitation of good regions
   - RL wanders around instead of refining near best feasible

2. **Reward Normalization Issues:**
   - Phase 2 reward: `3.0 × (improvement / (abs(old_best) + 1.0))`
   - For large objective values (~1000s), improvements get heavily normalized
   - Small improvements get tiny rewards → policy doesn't learn to optimize aggressively

3. **No Global Surrogate:**
   - qLogEI uses full GP on ALL data with proper kernels
   - RL only uses local RBF with 5 centers on last 50 points
   - RL is essentially "myopic" - doesn't see the full landscape

---

## Specific Evidence from Your Results

### Dimension 2 (easier):
- qLogEI: 78% feasible, avg -1165 when feasible
- RL: 47% feasible, avg -1484 when feasible
- **RL finds feasibility in only 42/90 runs vs qLogEI's 70/90**

### Dimension 10 (harder):
- qLogEI: 30% feasible, avg -1017 when feasible
- RL: 23% feasible, avg -3051 when feasible
- **RL gets MUCH worse in higher dimensions**

### Hardest Functions:
- F54: RL only 2/15 feasible (13%), qLogEI 3/15 (20%)
- F4: RL only 5/15 feasible (33%), qLogEI 9/15 (60%)

---

## What to Do for Your Poster

### Option 1: Be Honest (RECOMMENDED)

**Acknowledge the failure and analyze it:**

> *"Our meta-learned RL policy achieves only 35% feasibility rate compared to 54% for qLogEI,
> demonstrating that naive meta-learning without proper surrogate modeling fails on constrained BO.
> Analysis reveals three key issues: (1) inadequate reward shaping for feasibility-first optimization,
> (2) overly simple state representation lacking GP uncertainty, and (3) training-test mismatch in episode length."*

**This is GOOD for an academic poster:**
- Shows critical thinking
- Demonstrates you understand why it failed
- More valuable than cherry-picked good results

### Option 2: Focus on What You Learned

**Frame it as "What NOT to do in meta-learning for BO":**

1. ✗ Simple MLP policy without proper surrogate
2. ✗ Local RBF approximation vs full GP
3. ✗ Reward engineering that doesn't prioritize feasibility
4. ✓ qLogEI works well because it uses proper GPs + mathematical acquisition function

### Option 3: Compare Multiple RL Variants

I see you have `results_coco_rl_enhanced_vs_qlogei.csv` and `results_coco_rl_full_vs_qlogei.csv`.

Let me check if those are better:

---

## How to Fix This (Future Work Section)

### Fix 1: Better State Representation
- Include **full GP predictions** (μ, σ) at candidate points
- Add **constraint GP predictions** separately
- Include **expected improvement** and **probability of feasibility** as features

### Fix 2: Stronger Feasibility Rewards
- Massive penalty for remaining infeasible: `-10.0` per infeasible step
- Huge bonus for first feasible: `+50.0` instead of `+5.0`
- Progressive rewards: closer to feasible region = higher reward

### Fix 3: Hybrid Approach
- Use GP-based acquisition (like cEI) as a "proposal distribution"
- RL learns to SELECT among top-k cEI candidates
- This way RL refines a good baseline instead of learning from scratch

### Fix 4: Better Training Protocol
- Train with same budget as test time (10×D evaluations)
- Include harder functions during training
- Use curriculum learning: start with easy functions, progress to hard

### Fix 5: Use Transformer Instead
- Transformers can attend to full history
- No manual state feature engineering
- Can learn to mimic cEI behavior better

---

## Recommended Poster Structure Given These Results

### Section: Methods
- "We implemented an RL policy for meta-learning acquisition functions"
- Describe architecture, training, state representation

### Section: Results
- **Show the failure honestly**
- Table showing RL < qLogEI < Random (sometimes)
- Convergence plot showing RL struggles to find feasibility

### Section: Analysis (MOST IMPORTANT)
- "Why did meta-learning fail?"
  1. Inadequate surrogate modeling (local RBF vs full GP)
  2. Reward engineering challenges
  3. State representation limitations
  4. Training-test mismatch
- "What did we learn?"
  - Meta-learning requires careful integration with domain knowledge (GPs)
  - Naive RL without proper inductive bias fails
  - qLogEI works because it leverages Bayesian modeling

### Section: Future Work
- Hybrid RL + GP approach
- Better state features
- Transformer-based methods
- Curriculum learning

---

## Key Takeaway for Poster

**"Our results demonstrate that meta-learning for constrained BO is challenging:
naive RL policies without proper Bayesian surrogate modeling fail to match
traditional acquisition functions. This highlights the importance of integrating
domain knowledge (GPs, uncertainty quantification) into learned optimization policies."**

This is actually a **valuable contribution**: showing what doesn't work and why!

---

## Should You Include These Results?

**YES!** Negative results are scientifically valuable when properly analyzed.

**Famous examples:**
- Many ML papers show "baseline X doesn't work for reason Y"
- Science advances by understanding failures
- Your analysis of WHY it failed is the contribution

**How to frame it:**
- Don't say "our method works great" (it doesn't)
- Say "we investigated meta-learning for constrained BO and found..."
- Emphasize analysis over performance

---

## Next Steps

1. **Analyze the other result files** (enhanced, full) - are they better?
2. **Generate plots showing the failure** clearly
3. **Write honest analysis** in poster
4. **Focus on insights** not on claiming success

Want me to:
- Analyze the "enhanced" and "full" results to see if they're better?
- Create diagnostic plots showing where RL fails?
- Draft the poster text acknowledging these issues?
