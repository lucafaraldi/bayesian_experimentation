# Meta-Learning Acquisition Functions for Constrained Bayesian Optimization

**Course:** Bayesian Optimization 2025
**Institution:** [Your Institution]
**Authors:** [Your Names]

---

## Project Overview

This project explores **meta-learning approaches** to learn acquisition functions for constrained Bayesian Optimization. Instead of using hand-designed acquisition functions (like Expected Improvement or UCB), we train neural networks to suggest the next evaluation point by learning from experience across multiple optimization problems.

### Three Meta-Learning Approaches:

1. **Transformer-Based Next-Config Predictor**: Sequence-to-sequence model that predicts the next configuration given optimization history
2. **RL Policy Network (PPO)**: Reinforcement learning agent trained to maximize constrained optimization performance
3. **Learned Acquisition Function**: Deep neural network that mimics constrained Expected Improvement (cEI)

### Benchmark:
We evaluate our approaches against:
- **qLogEI (BoTorch)** - State-of-the-art constrained BO baseline (mandatory)
- **Random Search** - Naive baseline

on the **COCO BBOB-Constrained Suite** (Functions F2, F4, F6, F50, F52, F54).

---

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)

### Step 1: Clone the repository
```bash
git clone [your-repo-url]
cd bayesian_experimentation
```

### Step 2: Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch==2.2.2` - PyTorch for neural networks
- `botorch` - For qLogEI baseline (BoTorch library)
- `cocoex==2.8.1` - COCO experimentation framework
- `scikit-learn==1.7.2` - For Gaussian Processes in training
- `gymnasium==1.2.2` - For RL environment
- `matplotlib==3.10.7` - For plotting results

### Step 4: Verify COCO installation
```bash
python -c "import cocoex; print('COCO installed successfully!')"
```

---

## Repository Structure

```
bayesian_experimentation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── POSTER_CONTENT.md           # Complete poster content
├── POSTER_TODO.md              # Checklist for poster completion
│
├── models/                     # Trained models
│   ├── coco_policy_dim2.pt    # RL policy for D=2 ✓
│   └── coco_policy_dim10.pt   # RL policy for D=10 ✓
│
├── train_transformer.py        # Train Transformer next-config predictor
├── train_rl_new.py             # Train RL policy (PPO)
├── train_predictor.py          # Train learned acquisition function
├── benchmark_rl.py             # Benchmark all methods against qLogEI
├── plot_results.py             # Generate convergence plots
│
└── Practical Assignment BO 2025.pdf  # Assignment description
```

---

## Quick Start: Reproduce Results

### Option 1: Benchmark Pre-Trained RL Policy (Fastest)

We provide pre-trained RL policies for dimensions 2 and 10. To benchmark them against qLogEI and Random Search:

```bash
python benchmark_rl.py
```

**Outputs:**
- `results_coco_rl_vs_qlogei.csv` - Per-evaluation best feasible values
- `results_coco_rl_vs_qlogei_summary.csv` - Final results per repetition

**Expected runtime:** 2-4 hours (depending on hardware)

**Configuration:**
- Functions: F2, F4, F6, F50, F52, F54 (6 functions)
- Instances: 1, 2, 3 (3 instances each)
- Dimensions: 2, 10 (2 dimensions)
- Repetitions: 5 per instance
- Budget: 10 × D evaluations per run

**Total runs:** 6 functions × 3 instances × 2 dimensions × 5 repetitions × 3 methods = **540 optimization runs**

### Option 2: Generate Plots for Poster

After running benchmarks, generate convergence plots:

```bash
python plot_results.py
```

**Outputs in `figures/` directory:**
- `convergence_by_dimension.png` - Main convergence plot
- `convergence_by_function.png` - Per-function breakdown
- `performance_heatmap.png` - Performance across functions
- `final_performance_bars.png` - Bar chart comparison
- `summary_table.csv` - Numerical summary

---

## Training from Scratch

### 1. Train RL Policy (PPO)

```bash
python train_rl_new.py
```

**Configuration:**
- **Episodes:** 10,000 meta-training episodes
- **Horizon:** 15 RL steps per episode
- **Problem sampling:** Uniform over 18 (function, instance) pairs
- **Output:** `models/coco_policy_dim{D}.pt` for each dimension

**Training time:** ~4-6 hours per dimension (GPU recommended)

**Random seed:** Fixed at `GLOBAL_SEED = 123` in script

### 2. Train Transformer Next-Config Predictor

```bash
python train_transformer.py
```

**Configuration:**
- **Offline points:** 10,000 per COCO problem
- **Episodes:** 50 teacher trajectories per (function, instance) pair
- **Trajectory length:** T ∈ [20, 50]
- **Training samples:** ~100k per dimension
- **Output:** `next_config_transformer_dim{D}.pt`

**Training time:** ~6-10 hours per dimension

**Random seed:** `seed=42 + dim` in `main()`

### 3. Train Learned Acquisition Function

```bash
python train_predictor.py
```

**Configuration:**
- **Target samples:** 100k candidate evaluations per (function, instance) pair
- **Candidates per iteration:** 64 random points scored by teacher cEI
- **Teacher:** Constrained Expected Improvement (cEI) from GP surrogates
- **Output:** `learned_acquisition_cEI_dim{D}.pt`

**Training time:** ~6-10 hours per dimension

**Random seed:** `seed=42 + dim` in script

---

## Experimental Protocol

### Constrained BO Setup
- **Objective:** Minimize f(x) subject to g_j(x) ≤ 0
- **Aggregated constraint:** c(x) = max_j g_j(x)
- **Feasible region:** c(x) ≤ 0
- **Normalized space:** All x ∈ [0,1]^D

### Benchmark Protocol
For each method × problem × repetition:
1. **Initial design:** 2D random points in [0,1]^D
2. **Sequential optimization:** Remaining budget (10D - 2D) evaluations
3. **Tracking:** Best feasible objective value at each evaluation
4. **Reporting:** Final best feasible value and convergence curves

### Metrics
- **Primary:** Final best feasible value (lower is better)
- **Secondary:** Convergence speed, time to first feasible point
- **Statistical:** Mean ± std over 5 repetitions × 3 instances × 6 functions

---

## Random Seeds & Reproducibility

All experiments use **fixed random seeds** to ensure reproducibility:

### Benchmarking (`benchmark_rl.py`)
```python
# Seed per repetition
seed = 1234 + repetition_id
rng = np.random.default_rng(seed=seed)
```

### RL Training (`train_rl_new.py`)
```python
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
```

### Transformer Training (`train_transformer.py`)
```python
# In collect_data_for_dim():
np.random.seed(seed)
random.seed(seed)

# In train_next_config_model_for_dim():
seed = 42 + dim
```

### Learned Acquisition Training (`train_predictor.py`)
```python
# In train_learned_acquisition_for_dim():
np.random.seed(seed)
random.seed(seed)

# Called with:
seed = 42 + dim
```

**Important:** COCO problem evaluations themselves are deterministic given the same input point, so no additional seeding is needed for COCO.

---

## Modifying the Benchmarks

### Change budget or dimensions

Edit `benchmark_rl.py`:
```python
DIMENSIONS = [2, 10, 40]  # Add dimension 40
BUDGET_FACTOR = 20        # Change to 20*D evaluations
REPETITIONS = 3           # Reduce repetitions for faster testing
```

### Add Transformer to benchmark

1. Modify `benchmark_rl.py` to add a `TransformerOptimizer` class (similar to `RLOptimizer`)
2. Load model: `next_config_transformer_dim{dim}.pt`
3. Implement `suggest()` method that runs inference

### Add more COCO functions

Edit the `FUNCTIONS` list in any script:
```python
FUNCTIONS = [2, 4, 6, 8, 10, 50, 52, 54]  # Add F8, F10
```

---

## Key Design Choices

### Why these COCO functions?
- **F2, F4, F6:** Low-numbered functions (different difficulty)
- **F50, F52, F54:** High-numbered constrained functions
- Representative sample of the BBOB-constrained suite

### Why meta-learning?
- Hand-designed acquisition functions may not be optimal for all problems
- Meta-learning can discover problem-agnostic strategies
- Potential for better sample efficiency

### Why three approaches?
- **Transformer:** Natural for sequence prediction, no reward engineering
- **RL:** Direct optimization of performance, explicit exploration control
- **Learned Acq:** Mimics proven cEI strategy, fast inference

### Why constrained EI as teacher?
- Well-established baseline for constrained BO
- Balances improvement (EI) and feasibility (P(c≤0))
- Proven to work reasonably well

---

## Computational Requirements

### Benchmarking
- **CPU:** Intel i7 or equivalent
- **RAM:** 8 GB minimum
- **Time:** 2-4 hours for RL vs qLogEI vs Random on D={2,10}
- **Storage:** ~100 MB for result CSVs

### Training (per dimension)
- **GPU:** NVIDIA GPU with 8+ GB VRAM (recommended)
- **CPU fallback:** Possible but 5-10× slower
- **RAM:** 16 GB recommended
- **Time:**
  - RL: ~4-6 hours (GPU) / ~20-30 hours (CPU)
  - Transformer: ~6-10 hours (GPU) / ~30-50 hours (CPU)
  - Learned Acq: ~6-10 hours (GPU) / ~30-50 hours (CPU)
- **Storage:** ~100-500 MB per model

---

## Troubleshooting

### COCO import fails
```bash
pip install --upgrade coco-experiment cocopp
```

If still failing, install from source:
```bash
git clone https://github.com/numbbo/coco.git
cd coco
python do.py run-python
```

### BoTorch warnings
The code includes filters for common BoTorch warnings:
```python
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
```

These are cosmetic and don't affect results.

### GPU out of memory
Reduce batch size in training scripts:
```python
batch_size = 256  # Instead of 512
```

### Benchmark takes too long
Reduce problem set for testing:
```python
FUNCTIONS = [2, 6]        # Instead of all 6
INSTANCES = [1]           # Instead of [1,2,3]
REPETITIONS = 3           # Instead of 5
```

### Convergence issues in GP fitting
The scripts include convergence warning filters:
```python
warnings.filterwarnings("ignore", category=ConvergenceWarning)
```

If GPs fail to converge, results may be suboptimal but should still complete.

---

## Citation & Acknowledgments

This project was completed as part of the **Bayesian Optimization Course 2025**.

**COCO Benchmark:**
- Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software*, 36(1), 114-144.

**BoTorch:**
- Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. *Advances in Neural Information Processing Systems*, 33.

---

## Contact

For questions or issues, please contact:
- [Your Name]: [email]
- [Team Member 2]: [email]
- [Team Member 3]: [email]
- [Team Member 4]: [email]

**Repository:** [GitHub URL]

---

## License

This project is for educational purposes as part of the Bayesian Optimization Course 2025.
