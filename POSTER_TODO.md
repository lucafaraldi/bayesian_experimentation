# TODO Checklist for Poster Completion

## 1. Run Benchmarks ✓ PRIORITY

### What needs to be done:
- [ ] Run `benchmark_rl.py` to generate results comparing RL, qLogEI, and Random
- [ ] If you want to include Transformer results, modify `benchmark_rl.py` to add a Transformer optimizer
- [ ] If you want to include Learned Acquisition results, add that method too

### Expected outputs:
- `results_coco_rl_vs_qlogei.csv` - Per-evaluation best feasible values
- `results_coco_rl_vs_qlogei_summary.csv` - Final best feasible per repetition

### Command to run:
```bash
python benchmark_rl.py
```

**Estimated time:** 2-4 hours depending on hardware (you can reduce REPETITIONS to 3 or DIMENSIONS to [2] for faster testing)

---

## 2. Generate Plots for Poster ✓ PRIORITY

### What needs to be created:

#### A. Convergence Plots (REQUIRED by assignment)
- [ ] Plot best feasible value vs. number of evaluations
- [ ] One plot per dimension (D=2, D=10)
- [ ] Show all methods: Random, qLogEI, RL (optionally Transformer)
- [ ] Include error bars or shaded regions (±1 std)
- [ ] Average across all functions and instances, OR
- [ ] Show 2-3 representative functions (e.g., F2, F6, F50)

#### B. Summary Table
- [ ] Final best feasible values: mean ± std
- [ ] Broken down by dimension
- [ ] Statistical significance tests (optional but good)

#### C. Additional Analysis (optional but impressive)
- [ ] Performance heatmap by function ID
- [ ] Bar chart comparing methods
- [ ] Time-to-feasibility plot (how many evals to find first feasible point)

### Suggested tool:
```bash
# Create a plot_results.py script using matplotlib/seaborn
python plot_results.py
```

---

## 3. Complete Missing Results Sections in POSTER_CONTENT.md

### Fill in the placeholders:
- [ ] Replace "X.XXe+XX" with actual mean ± std values from results
- [ ] Add "Overall Rank" column (1st, 2nd, 3rd, 4th)
- [ ] Add key observations about which method performs best
- [ ] Add specific numbers: "X% improvement over qLogEI", "Y% faster convergence"

---

## 4. Optional: Train Transformer Models

### If you want to include Transformer in benchmarks:
- [ ] Run `train_transformer.py` for dim=2 and dim=10
- [ ] This will generate `next_config_transformer_dim2.pt` and `next_config_transformer_dim10.pt`
- [ ] Modify `benchmark_rl.py` to add a TransformerOptimizer class
- [ ] Re-run benchmarks

**Estimated time:** Several hours (transformer training is slow)

---

## 5. Optional: Train Learned Acquisition Models

### If you want to include Learned Acquisition in benchmarks:
- [ ] Run `train_predictor.py` for dim=2 and dim=10
- [ ] This will generate `learned_acquisition_cEI_dim2.pt` and `learned_acquisition_cEI_dim10.pt`
- [ ] Modify `benchmark_rl.py` to add a LearnedAcqOptimizer class
- [ ] Re-run benchmarks

**Estimated time:** Several hours

---

## 6. Create Poster Visual Design

### Options:
1. **PowerPoint/Keynote** (easiest)
   - Use A0 landscape template (119.4 cm × 84.1 cm)
   - Copy content from POSTER_CONTENT.md
   - Insert plots and tables
   - Export as PDF

2. **LaTeX with beamerposter** (professional)
   - Use beamer poster template
   - Include TikZ diagrams for architecture figures
   - Programmatic inclusion of result figures

3. **Scientific poster tools**
   - Canva (online, templates available)
   - Adobe Illustrator
   - Inkscape (free)

### Layout suggestion:
```
+------------------------------------------------------------------+
|  TITLE: Meta-Learning Acquisition Functions for Constrained BO  |
|  Authors | Institution | Course                                  |
+------------------------------------------------------------------+
|                    |                      |                      |
|   1. MOTIVATION    |   3. METHODS         |   5. RESULTS         |
|   & PROBLEM        |   ---------------    |   ---------------    |
|   ---------------  |   - Transformer      |   - Convergence plots|
|   - Why meta-      |   - RL Policy        |   - Summary table    |
|     learning?      |   - Learned Acq      |   - Analysis         |
|   - COCO setup     |   [Architecture      |   [FIGURES]          |
|                    |    diagrams]         |                      |
|                    |                      |                      |
|   2. BASELINES     |   4. EXPERIMENTAL    |   6. CONCLUSIONS     |
|   ---------------  |      SETUP           |   & FUTURE WORK      |
|   - qLogEI         |   ---------------    |   ---------------    |
|   - Random         |   - Budget: 10D      |   - Key insights     |
|                    |   - Functions: F2-54 |   - Limitations      |
|                    |   - 5 repetitions    |   - Future work      |
+------------------------------------------------------------------+
|                    REFERENCES & CODE                             |
+------------------------------------------------------------------+
```

---

## 7. Create README.md (REQUIRED by assignment)

### Must include:
- [ ] Project title and description
- [ ] Installation instructions
  - Python version
  - `pip install -r requirements.txt`
  - COCO installation notes if needed
- [ ] How to run experiments
  - Training commands for each method
  - Benchmarking command
  - Plotting command
- [ ] Repository structure explanation
- [ ] Random seed documentation
- [ ] Expected outputs and where to find them

---

## 8. Prepare for Presentation

### Practice explaining:
- [ ] **Motivation:** Why meta-learning? (30 seconds)
- [ ] **Methods:** High-level overview of 3 approaches (1 minute)
- [ ] **Results:** Key findings from convergence plots (30 seconds)
- [ ] **Analysis:** What worked, what didn't, why? (1 minute)

### Anticipate questions:
- "Why did you choose these specific COCO functions?"
- "How does computational cost compare between methods?"
- "Did you try combining methods?"
- "What about dimension 40?"
- "How do you ensure your RL policy doesn't overfit?"

---

## 9. Final Checks Before Submission

- [ ] Poster is A0 landscape (119.4 × 84.1 cm)
- [ ] All figures are high resolution (300 DPI minimum)
- [ ] Convergence plots clearly show all methods with legend
- [ ] Text is readable from 1 meter away (minimum 24pt font)
- [ ] References are included
- [ ] Code repository is clean and documented
- [ ] README.md is complete with reproducibility instructions
- [ ] requirements.txt is up to date
- [ ] Poster PDF submitted before deadline

---

## Priority Order (if short on time)

### MUST DO:
1. Run benchmark_rl.py (just RL vs qLogEI vs Random)
2. Generate convergence plots
3. Fill in results in poster content
4. Create poster visual (PowerPoint is fastest)
5. Write README.md

### NICE TO HAVE:
6. Train and benchmark Transformer
7. Train and benchmark Learned Acquisition
8. Create additional analysis plots
9. Statistical significance tests
10. Fancy LaTeX poster

---

## Time Estimates

| Task | Time | Priority |
|------|------|----------|
| Run benchmarks (RL only) | 2-4 hours | HIGH |
| Generate plots | 1 hour | HIGH |
| Fill in results | 30 min | HIGH |
| Create poster visual | 3-4 hours | HIGH |
| Write README | 1 hour | HIGH |
| Train Transformer | 4-6 hours | MEDIUM |
| Train Learned Acq | 4-6 hours | MEDIUM |
| Additional analysis | 2 hours | LOW |
| **TOTAL (must-do)** | **8-11 hours** | - |
| **TOTAL (with optional)** | **18-25 hours** | - |

---

## Notes

- You already have trained RL models (coco_policy_dim2.pt, coco_policy_dim10.pt) ✓
- benchmark_rl.py is already written and ready to run ✓
- Focus on getting SOLID results for RL vs qLogEI vs Random first
- You can add Transformer/Learned Acq results later if time permits
- The poster content document (POSTER_CONTENT.md) has all the text you need

**Remember:** It's better to have complete results for fewer methods than incomplete results for all methods!
