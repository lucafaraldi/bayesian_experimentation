# Quick Start Guide

## What You Have Now ✓

✅ **Code Structure:**
- 3 meta-learning approaches implemented:
  - Transformer-based next-config predictor
  - RL policy network (PPO)
  - Learned acquisition function
- Benchmarking script ready to run
- Plotting script ready to generate figures

✅ **Trained Models:**
- `models/coco_policy_dim2.pt` - RL policy for D=2
- `models/coco_policy_dim10.pt` - RL policy for D=10

✅ **Documentation:**
- `POSTER_CONTENT.md` - Complete poster text
- `POSTER_TODO.md` - Checklist for completion
- `README.md` - Full documentation for reproducibility

---

## Next Steps to Complete the Poster

### STEP 1: Run Benchmarks (2-4 hours) ⭐ PRIORITY

```bash
python benchmark_rl.py
```

This will:
- Compare RL vs qLogEI vs Random Search
- Test on all 6 COCO functions × 3 instances × 2 dimensions
- Run 5 repetitions per configuration
- Generate CSV files with results

**Output:**
- `results_coco_rl_vs_qlogei.csv`
- `results_coco_rl_vs_qlogei_summary.csv`

**Tip:** Open another terminal and run `watch -n 10 ls -lh results_*.csv` to monitor progress.

---

### STEP 2: Generate Plots (5 minutes) ⭐ PRIORITY

Once benchmarks are done:

```bash
python plot_results.py
```

This will create:
- `figures/convergence_by_dimension.png` - Main convergence plot (REQUIRED for poster)
- `figures/convergence_by_function.png` - Per-function breakdown
- `figures/performance_heatmap.png` - Performance across functions
- `figures/final_performance_bars.png` - Bar chart comparison
- `figures/summary_table.csv` - Numerical results

---

### STEP 3: Update Poster Content (30 minutes) ⭐ PRIORITY

1. Open `POSTER_CONTENT.md`
2. In **Section 5 (RESULTS)**, fill in the summary table with actual numbers from `figures/summary_table.csv`
3. Add key observations:
   - "RL achieves X% improvement over qLogEI in D=2"
   - "qLogEI converges faster in D=10 but RL achieves better final performance"
   - (or whatever your results show!)

---

### STEP 4: Create Poster Visual (3-4 hours) ⭐ PRIORITY

**Option A: PowerPoint (Easiest)**
1. Create new presentation with A0 landscape slide (119.4 cm × 84.1 cm)
2. Copy sections from `POSTER_CONTENT.md`
3. Insert figures from `figures/` directory
4. Use layout from `POSTER_CONTENT.md` (end of document)
5. Export as PDF

**Option B: LaTeX beamerposter (More professional)**
1. Use template from Overleaf: "Academic Poster"
2. Import content and figures
3. Compile to PDF

**Color Scheme:**
- Blue = qLogEI
- Red = Random
- Green = RL
- Orange = Transformer (if you add it)

---

### STEP 5: Update README (30 minutes)

1. Fill in your names, institution, emails
2. Add actual training times from your experiments
3. Add your GitHub repository URL

---

## Optional Enhancements (If You Have Time)

### Add Transformer to Benchmark

1. Train transformer models:
```bash
python train_transformer.py  # ~6-10 hours
```

2. Modify `benchmark_rl.py` to add `TransformerOptimizer` class

3. Re-run benchmarks and plots

### Add Learned Acquisition to Benchmark

1. Train learned acquisition models:
```bash
python train_predictor.py  # ~6-10 hours
```

2. Modify `benchmark_rl.py` to add `LearnedAcqOptimizer` class

3. Re-run benchmarks and plots

### Statistical Significance Tests

Add to `plot_results.py`:
```python
from scipy import stats

# Wilcoxon signed-rank test
rl_final = summary_df[summary_df['method']=='RL']['final_best_feasible']
qlogei_final = summary_df[summary_df['method']=='qLogEI']['final_best_feasible']
statistic, p_value = stats.wilcoxon(rl_final, qlogei_final)
print(f"P-value: {p_value:.4f}")
```

---

## Time Budget

| Task | Time | Status |
|------|------|--------|
| Run benchmarks | 2-4 hours | ⏳ TODO |
| Generate plots | 5 minutes | ⏳ TODO |
| Fill in results | 30 minutes | ⏳ TODO |
| Create poster visual | 3-4 hours | ⏳ TODO |
| Update README | 30 minutes | ⏳ TODO |
| **TOTAL MINIMUM** | **7-10 hours** | - |

**With optional:**
| Add Transformer | 8-12 hours | Optional |
| Add Learned Acq | 8-12 hours | Optional |

---

## Troubleshooting

### "No module named 'botorch'"
```bash
pip install botorch gpytorch
```

### "results_coco_rl_vs_qlogei.csv not found"
Run `benchmark_rl.py` first!

### Benchmark is too slow
Edit `benchmark_rl.py`:
```python
FUNCTIONS = [2, 6]     # Test with fewer functions
INSTANCES = [1]        # Test with one instance
REPETITIONS = 3        # Reduce repetitions
```

### Want to test quickly
```python
DIMENSIONS = [2]       # Only dimension 2
BUDGET_FACTOR = 5      # Reduce budget to 5*D
```

---

## What Makes a Good Poster?

✅ **Clear story:** Motivation → Methods → Results → Conclusions
✅ **Visual hierarchy:** Large title, clear sections, readable from 1 meter
✅ **Good figures:** High-resolution, clear labels, legend visible
✅ **Convergence plot:** Shows all methods, with error bars (REQUIRED)
✅ **Key numbers:** Highlight improvements: "+X% over baseline"
✅ **Honest analysis:** Discuss what worked AND what didn't
✅ **Future work:** Show you understand limitations

❌ **Avoid:** Walls of text, tiny fonts, unclear plots, missing baselines

---

## Presentation Tips

**Prepare 2-minute pitch:**
1. **Hook (15s):** "Can we learn better acquisition functions from data?"
2. **Problem (20s):** "Traditional BO uses hand-designed heuristics like EI"
3. **Solution (30s):** "We trained neural networks on COCO problems using 3 approaches"
4. **Results (30s):** "Our RL policy achieves X% improvement over qLogEI baseline"
5. **Takeaway (15s):** "Meta-learning is promising but challenges remain in scaling"

**Common questions:**
- "Why these 3 approaches?" → Different tradeoffs (sequence modeling vs RL vs supervised)
- "What about dimension 40?" → Challenge: state space grows, models trained on D=2,10
- "Training cost?" → RL: ~4-6 hours/dim, worth it if deployed on many problems
- "Generalization?" → Tested on same function classes; future work: held-out functions

---

## Files You Need to Submit

📄 **Poster PDF** (A0 landscape, digital version)
📄 **Code repository** (GitHub or zip)
  - All .py files
  - requirements.txt
  - README.md
  - Trained models in models/
  - Results CSVs (optional but good to include)

---

## Final Checklist Before Submission

- [ ] Poster has convergence plots (REQUIRED)
- [ ] All methods shown: Random, qLogEI, RL (minimum)
- [ ] Results table filled in with actual numbers
- [ ] README.md has complete installation instructions
- [ ] requirements.txt includes all dependencies
- [ ] Code runs with fixed random seeds
- [ ] Poster is A0 landscape format
- [ ] Your names are on the poster
- [ ] References included

---

## Good Luck! 🎯

You have everything you need:
- ✅ Working code
- ✅ Trained models
- ✅ Complete documentation
- ✅ Poster content written

Just run the benchmarks, generate the plots, fill in the numbers, and create the visual design.

**Estimated total time to finish:** 7-10 hours
**When to start:** As soon as possible! (benchmarks take 2-4 hours)

---

## Questions?

Check:
1. `POSTER_TODO.md` - Detailed checklist
2. `README.md` - Full documentation
3. `POSTER_CONTENT.md` - Complete poster text

Still stuck? Review the assignment PDF: `Practical Assignment BO 2025.pdf`
