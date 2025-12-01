# Meta-Learning Acquisition Functions for Constrained Bayesian Optimization
## Poster Content Structure (A0 Landscape)

---

## 1. MOTIVATION & NOVELTY

### Why Meta-Learning for Acquisition Functions?

**The Challenge:**
- Traditional acquisition functions (EI, UCB, qLogEI) are hand-designed heuristics
- They may not be optimal across diverse problem landscapes
- Constrained BO adds complexity: need to balance exploration, exploitation, AND feasibility

**Our Innovation:**
Instead of using fixed acquisition functions, we **learn** acquisition strategies from data across multiple constrained optimization problems.

**Key Insight:**
By training on a diverse set of COCO constrained BBOB functions, we can capture patterns that generalize to new optimization landscapes.

**Why This Matters:**
- Potential for better sample efficiency (fewer evaluations needed)
- Adaptive behavior learned from experience
- Can discover non-obvious exploration strategies

---

## 2. PROBLEM SETUP

### Constrained Bayesian Optimization on COCO BBOB-Constrained Suite

**Objective:** Minimize f(x) subject to g_j(x) ≤ 0 for j = 1,...,m

**Aggregated Constraint:** c(x) = max_j g_j(x)
- Feasible region: c(x) ≤ 0

**Benchmark Suite:**
- **Functions:** F2, F4, F6, F50, F52, F54 (6 functions)
- **Instances:** 1, 2, 3 (3 instances per function)
- **Dimensions:** 2, 10 (extensible to 40)
- **Budget:** 10 × D evaluations
- **Repetitions:** 5 per instance

**Total Test Problems:** 6 functions × 3 instances × 2 dimensions = 36 problem variants

---

## 3. APPROACHES: THREE META-LEARNING STRATEGIES

### 3.1 Transformer-Based Next-Config Predictor

**Architecture:**
```
Input: History sequence (x₁, y₁, c₁), ..., (xₜ, yₜ, cₜ)
Model: Transformer Encoder with positional encoding
Output: Next configuration xₜ₊₁
```

**Training Strategy:**
- **Teacher:** Constrained EI (cEI) from Gaussian Process surrogates
- **Student:** Transformer learns to predict teacher's choices
- **Data Collection:**
  - 10,000 offline points per COCO problem
  - Episodes with T ∈ [20, 50] trajectory length
  - 50 episodes per (function, instance) pair
  - ~100k training samples per dimension

**Key Features:**
- 4-head attention with 3 encoder layers
- Token representation: [x (D-dim), y (1), c (1)]
- Learns sequence-to-sequence mapping like "next word prediction"
- Captures temporal dependencies in optimization trajectories

**Model Details:**
- d_model = 128, hidden_dim = 256
- Trained for 50 epochs with AdamW (lr=1e-3)
- MSE loss on predicted next configuration

---

### 3.2 RL Policy Network (PPO-Based)

**Architecture:**
```
State → [Summary Stats + Surrogate Features + Last Point + Best Feasible Point]
Policy Network (Gaussian) → Action (next point in [0,1]ᴰ)
Value Network → State value estimate
```

**State Representation (13 + 2D dimensions):**
1. **Summary Statistics (10 dims):**
   - Best feasible value (so far)
   - Mean/std of observed objectives
   - Number and ratio of feasible points
   - Current step, horizon, dimension

2. **Local Surrogate Features (3 dims):**
   - RBF regression predictions at last point
   - RBF prediction at best feasible point
   - Residual standard deviation

3. **Geometric Features (2D dims):**
   - Last evaluated point (D dims)
   - Best feasible point found (D dims)

**Training:**
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Episodes:** 10,000 meta-training episodes
- **Horizon:** 15 RL steps per episode
- **Initial Design:** 4D random points before RL starts
- **Problem Sampling:** Uniformly sample from 18 (function, instance) pairs

**Reward Design (Feasibility-First):**
- **Phase 1 (no feasible yet):**
  - +5.0 bonus for first feasible point
  - Penalty for constraint violation
  - +0.1 novelty bonus (distance to previous points)

- **Phase 2 (after feasibility):**
  - +3.0 × normalized improvement in best feasible
  - Light penalty for infeasible suggestions
  - +0.2 novelty bonus

**Policy Network:**
- 2-layer MLP (256 hidden units, Tanh activation)
- Gaussian output with learnable log_std
- Action mapped to [0,1]ᴰ via tanh → rescale

---

### 3.3 Learned Acquisition Function (Deep Neural Network)

**Architecture:**
```
Candidate Features φ(Hₜ, x) + Global Features g(Hₜ) → Deep MLP → Acquisition Score
```

**Feature Engineering:**

1. **Candidate Features φ(Hₜ, x) (D + 10 dimensions):**
   - Raw location x ∈ [0,1]ᴰ
   - GP predictions: μf(x), σf(x), μc(x), σc(x)
   - Probability of feasibility: P(c(x) ≤ 0)
   - Current best feasible value
   - Expected improvement: f_best - μf(x)
   - Distance to best feasible point
   - Min distance to feasible/infeasible regions

2. **Global Features g(Hₜ) (8 dimensions):**
   - Problem dimension D
   - Optimization progress (t/budget)
   - Current best feasible value
   - Fraction of feasible points
   - Mean/std of observed objectives
   - Mean/std of constraint violations

**Network Architecture:**
- Input: φ(Hₜ, x) ⊕ g(Hₜ)
- 5-layer MLP: 256 → 256 → 128 → 64 → 1
- GELU activations, LayerNorm after first two layers
- Output: Scalar score approximating cEI(x)

**Training:**
- **Teacher:** Constrained Expected Improvement (cEI)
  - cEI(x) = EI(x) × P(feasible)
  - EI from GP surrogate on objectives
  - P(feasible) from GP on aggregated constraints

- **Data Collection:**
  - ~100k candidate evaluations per (function, instance) pair
  - At each BO iteration: 64 random candidates scored by teacher
  - Supervised learning: minimize MSE between student and teacher scores

- **Optimization:**
  - 50-200 epochs with AdamW (lr=1e-3, weight_decay=1e-5)
  - Batch size: 512
  - 90/10 train/val split

---

## 4. EXPERIMENTAL SETUP

### Baselines

1. **qLogEI (BoTorch)** - Mandatory State-of-the-Art Baseline
   - Log Expected Improvement with constraint handling
   - Batch size q=1
   - ModelListGP: separate GPs for objective and constraint
   - MC sampling with 128 samples
   - 5 restarts for acquisition optimization

2. **Random Search** - Naive Baseline
   - Uniform random sampling in [0,1]ᴰ
   - No exploitation, pure exploration

### Evaluation Protocol

**For each method × problem × repetition:**
1. Initial random design: 2D points
2. Sequential optimization: remaining budget (10D - 2D evaluations)
3. Track best feasible value at each iteration
4. Report final performance and convergence curves

**Metrics:**
- **Primary:** Best feasible objective value at budget exhaustion
- **Secondary:** Number of evaluations to reach feasibility
- **Analysis:** Convergence plots showing best feasible vs. evaluations

---

## 5. RESULTS

### Summary Statistics

**[TABLE: Final Best Feasible Values]**
```
Method          | Dim=2 (Mean ± Std)  | Dim=10 (Mean ± Std) | Overall Rank
----------------|---------------------|---------------------|-------------
Random          | X.XXe+XX ± X.XXe+XX | X.XXe+XX ± X.XXe+XX | 4
qLogEI          | X.XXe+XX ± X.XXe+XX | X.XXe+XX ± X.XXe+XX | ?
RL Policy       | X.XXe+XX ± X.XXe+XX | X.XXe+XX ± X.XXe+XX | ?
Transformer     | X.XXe+XX ± X.XXe+XX | X.XXe+XX ± X.XXe+XX | ?
```

**Note:** Lower is better (minimization). Averaged over all functions, instances, and repetitions.

### Convergence Plots

**[FIGURE 1: Convergence on Representative Problems]**
- 2×3 grid: 2 dimensions × 3 representative functions (e.g., F2, F6, F50)
- X-axis: Number of evaluations
- Y-axis: Best feasible objective value (log scale)
- Lines: Random (red), qLogEI (blue), RL (green), Transformer (orange)
- Shaded regions: ±1 std deviation across repetitions

**Key Observations to Highlight:**
- Which method converges fastest?
- Which achieves best final performance?
- Are meta-learned methods better than qLogEI?
- Performance difference in low vs. high dimensions?

### Performance by Function Class

**[FIGURE 2: Heatmap or Bar Chart]**
- Show final performance breakdown by function ID
- Identify which functions favor meta-learning vs. traditional methods

---

## 6. ANALYSIS & DISCUSSION

### What Did We Learn?

**Meta-Learning Advantages:**
- [If positive results] Meta-learned policies show X% improvement over qLogEI
- Faster convergence to feasibility in [specific cases]
- Better exploration-exploitation balance in [specific scenarios]

**Challenges & Limitations:**
- Training cost: 10k episodes × 18 problems = significant compute
- Generalization: Do policies trained on F{2,4,6,50,52,54} work on unseen functions?
- Dimension scaling: Promising in D=2,10 but D=40 remains challenging
- Transformer memory: Long sequences (T=50) require careful architecture design

### Comparison of Meta-Learning Approaches

| Approach       | Pros | Cons |
|----------------|------|------|
| **Transformer**| - Natural sequence modeling<br>- Can capture long-term dependencies<br>- No reward engineering | - Slow inference (attention overhead)<br>- Requires large trajectory datasets<br>- Less exploration control |
| **RL Policy**  | - Direct reward optimization<br>- Novelty bonus encourages exploration<br>- Fast inference | - Reward engineering needed<br>- Sensitive to hyperparameters<br>- Training instability |
| **Learned Acq**| - Fast candidate scoring<br>- Mimics proven cEI strategy<br>- Easy to integrate | - Limited to teacher's knowledge<br>- Feature engineering required<br>- No explicit exploration mechanism |

### Future Directions

1. **Scaling to Higher Dimensions:**
   - Investigate dimension-independent state representations
   - Use embeddings or attention mechanisms for variable-D inputs
   - Train on D=40 problems with increased compute budget

2. **Cross-Function Generalization:**
   - Test on held-out COCO functions (F8, F10, ...)
   - Evaluate on real-world constrained optimization problems
   - Meta-learning with domain adaptation

3. **Hybrid Approaches:**
   - Combine learned acquisition with traditional methods
   - Warm-start with qLogEI, then switch to learned policy
   - Ensemble methods: vote among transformer, RL, and qLogEI

4. **Online Adaptation:**
   - Fine-tune learned policies during optimization
   - Few-shot learning for new problem classes
   - Contextual bandits for acquisition function selection

5. **Multi-Objective Extensions:**
   - Extend to multi-objective constrained BO
   - Learn Pareto-optimal acquisition strategies

6. **Interpretability:**
   - Visualize attention weights in transformer
   - Analyze learned features in acquisition network
   - Compare decision boundaries with analytical acquisition functions

---

## 7. CONCLUSION

### Key Contributions

1. **Novel Framework:** First systematic study of meta-learning for constrained BO acquisition functions on standardized COCO benchmarks

2. **Three Approaches:** Implemented and compared Transformer, RL, and learned acquisition networks—each with unique strengths

3. **Rigorous Evaluation:** Comprehensive benchmarking against qLogEI baseline on 36 problem variants with proper experimental protocol

4. **Open Questions:** Identified key challenges (scaling, generalization, training cost) and promising future directions

### Takeaway Message

Meta-learning acquisition functions represents a promising direction for constrained Bayesian Optimization, offering potential improvements over hand-crafted heuristics by learning from diverse problem experiences. While challenges remain in scaling and generalization, our results demonstrate [positive/mixed/promising] outcomes and open paths for future research.

---

## REFERENCES

1. BoTorch tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only/
2. COCO Constrained Suite: https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf
3. Constrained BO benchmark paper: https://arxiv.org/pdf/2506.14619

---

## IMPLEMENTATION DETAILS

**Code Repository:** [Your GitHub URL]

**Key Files:**
- `train_transformer.py`: Transformer-based next-config predictor
- `train_rl_new.py`: RL policy training with PPO
- `train_predictor.py`: Learned acquisition function training
- `benchmark_rl.py`: Benchmarking script for all methods
- `requirements.txt`: Dependencies (PyTorch, BoTorch, COCO, scikit-learn)

**Reproducibility:**
- Fixed random seeds throughout
- All hyperparameters documented in code
- Trained models available in `models/` directory

**Computational Requirements:**
- RL training: ~X hours on [GPU type]
- Transformer training: ~Y hours on [GPU type]
- Benchmarking: ~Z hours on [CPU/GPU]

---

## ACKNOWLEDGMENTS

We thank [course instructor name] for guidance and the COCO team for providing the constrained benchmark suite.

---

**NOTES FOR POSTER DESIGN:**
- Use color coding: Blue=qLogEI, Red=Random, Green=RL, Orange=Transformer
- Include 2-3 key figures: convergence plots, performance table, method comparison diagram
- Keep text concise: bullet points over paragraphs
- Visual flow: Motivation → Problem → Methods → Results → Conclusions
- Highlight key numbers: X% improvement, Y fewer evaluations, etc.
