# Offline vs Online: The Critical Difference

## The Core Problem

**Transformer was trained OFFLINE but tested ONLINE.**

This mismatch is like:
- Training a self-driving car in a simulator, then testing on real roads
- Learning surgery from textbooks, then operating on real patients
- Practicing chess against yourself, then playing in a tournament

---

## What is "Offline"?

**Offline = Pre-computed, fixed dataset, no real interaction**

### How Transformer Training Works (Offline):

```
BEFORE any optimization starts:
├─ Step 1: Sample 10,000 random points
│  └─ Evaluate f(x) and constraints g(x) for all 10k points
│  └─ This is EXPENSIVE but done once per problem
│
├─ Step 2: Build teacher episodes FROM this fixed data
│  └─ Teacher (cEI) has access to ALL 10k evaluations
│  └─ Teacher selects trajectory by picking from this pool
│  └─ Example episode: points [3,157, 8,421, 1,092, ...]
│  └─ Generate 50 episodes per problem
│
└─ Step 3: Train Transformer on these episodes
   └─ Input: prefix of episode (first t points)
   └─ Output: next point in episode (point t+1)
   └─ The Transformer NEVER evaluates anything new
```

### Key Characteristics of Offline Learning:

| Aspect | Offline Training |
|--------|------------------|
| **Data availability** | ALL 10k points available upfront |
| **Information** | Complete: see f(x), g(x) for all 10k points |
| **Cost** | Front-loaded: 10k evaluations before learning |
| **Exploration** | Already done (10k random points cover space) |
| **Teacher's job** | SELECT from existing pool |
| **Risk** | Zero (no new evaluations = no cost) |

---

## What is "Online"?

**Online = Sequential, expensive evaluations, learn as you go**

### How Testing Works (Online):

```
START with nothing:
├─ Step 1: Initial random design (4 points for D=2)
│  └─ Evaluate f(x₁), g(x₁), f(x₂), g(x₂), f(x₃), g(x₃), f(x₄), g(x₄)
│  └─ Budget used: 4/20
│
├─ Step 2: Transformer suggests x₅ based on history
│  └─ Input: (x₁,y₁,c₁), (x₂,y₂,c₂), (x₃,y₃,c₃), (x₄,y₄,c₄)
│  └─ Output: x₅ (prediction)
│  └─ MUST EVALUATE: f(x₅), g(x₅) ← EXPENSIVE!
│  └─ Budget used: 5/20
│
├─ Step 3: Transformer suggests x₆ based on updated history
│  └─ Input: (x₁,y₁,c₁), ..., (x₅,y₅,c₅)
│  └─ Output: x₆ (prediction)
│  └─ MUST EVALUATE: f(x₆), g(x₆) ← EXPENSIVE!
│  └─ Budget used: 6/20
│
└─ Continue until budget exhausted (20 evaluations total)
```

### Key Characteristics of Online Optimization:

| Aspect | Online Testing |
|--------|----------------|
| **Data availability** | ONLY what you've evaluated so far |
| **Information** | Incomplete: see f(x), g(x) only at chosen points |
| **Cost** | Per-step: each evaluation counts against budget |
| **Exploration** | Must be strategic (limited budget!) |
| **Optimizer's job** | DECIDE which point to evaluate next |
| **Risk** | High (bad choice wastes precious budget) |

---

## Side-by-Side Comparison

### Example: D=2, Budget=20 evaluations

#### OFFLINE (Training):

```
Problem: F2, Instance 1, D=2

Phase 1 - Data Collection (done once):
  Evaluate 10,000 random points:
  x₁ = [0.15, 0.82] → f=1523, g=0.3  (infeasible)
  x₂ = [0.91, 0.23] → f=892, g=-0.1  (feasible) ✓
  x₃ = [0.45, 0.67] → f=1204, g=0.15 (infeasible)
  ... [9,997 more points] ...

  This costs 10,000 evaluations but gives complete picture!

Phase 2 - Teacher creates episode:
  Teacher (cEI) looks at ALL 10k points
  Teacher picks best trajectory:
    t=1: Look at 10k points, pick x₂ (high cEI)
    t=2: Look at 10k points, pick x₇₃₄ (high cEI)
    t=3: Look at 10k points, pick x₁₉₈ (high cEI)
    ...
    t=20: Look at 10k points, pick x₅₂₁₁ (high cEI)

  Teacher has perfect information at every step!

Phase 3 - Transformer training:
  Learn to predict: given [x₂, x₇₃₄], predict x₁₉₈
  Learn to predict: given [x₂, x₇₃₄, x₁₉₈], predict x₅₂₁₁
  ...

  Transformer NEVER evaluates anything—just learns patterns!
```

#### ONLINE (Testing):

```
Problem: F6, Instance 2, D=2 (DIFFERENT PROBLEM!)

Start: Budget = 20, History = empty

Step 1-4 (Initial design):
  x₁ = [0.23, 0.71] → EVALUATE → f=2341, g=0.5 (infeasible)
  x₂ = [0.67, 0.19] → EVALUATE → f=1842, g=0.2 (infeasible)
  x₃ = [0.88, 0.92] → EVALUATE → f=3021, g=-0.05 (feasible) ✓
  x₄ = [0.11, 0.43] → EVALUATE → f=2124, g=0.8 (infeasible)

  Budget used: 4/20
  Transformer knows: only these 4 points!

Step 5:
  Transformer input: [(x₁,y₁,c₁), (x₂,y₂,c₂), (x₃,y₃,c₃), (x₄,y₄,c₄)]
  Transformer output: x₅ = [0.81, 0.87] (next guess)

  MUST EVALUATE x₅ → f=2890, g=0.05 (infeasible)

  Budget used: 5/20
  If x₅ is bad, we wasted 1/20 of our budget!

Step 6:
  Transformer input: [(x₁,y₁,c₁), ..., (x₅,y₅,c₅)]
  Transformer output: x₆ = [0.75, 0.81]

  MUST EVALUATE x₆ → f=2654, g=-0.02 (feasible) ✓

  Budget used: 6/20

... Continue until budget exhausted ...

Final result: Best feasible found = f(x₃) = 3021
  (Compare to qLogEI: best = 412)

Transformer had NO offline data, NO 10k points, just 20 evaluations!
```

---

## Why This Mismatch Matters

### Problem 1: Information Asymmetry

**Training (Offline):**
- Teacher sees 10,000 evaluated points
- Can compare candidates: "x₁₂₃ has f=500, x₄₅₆ has f=800, pick x₁₂₃"
- Has dense coverage of space
- GP trained on 10k points = accurate

**Testing (Online):**
- Transformer sees only 4-6 evaluated points initially
- Cannot compare: "no idea what's at x₁₂₃ or x₄₅₆, must guess"
- Sparse coverage of space
- No GP, just pattern matching from training

**Analogy:** Like training a doctor by showing them 10,000 patients' complete medical histories, then asking them to diagnose someone with only 4 symptoms and no tests.

---

### Problem 2: Exploration Strategy

**Training (Offline):**
```python
# Teacher's selection logic
candidates = all_10k_points  # Already evaluated!
for x in candidates:
    cEI[x] = compute_cEI(x, history)  # Fast computation
best = argmax(cEI)  # Pick best from 10k options
return best  # No evaluation needed!
```
- Exploration already done (10k random points)
- Teacher just EXPLOITS (picks best from pool)
- No exploration-exploitation tradeoff

**Testing (Online):**
```python
# Transformer's decision
history = [(x₁,y₁,c₁), ..., (x_t,y_t,c_t)]  # Only t points!
x_next = transformer.predict(history)  # Guess based on pattern
evaluate(x_next)  # EXPENSIVE! Uses 1/20 budget!
```
- Must explore AND exploit with limited budget
- Each evaluation is precious
- Transformer doesn't know where to explore (never learned that!)

**Analogy:** Training a treasure hunter by showing them maps with all treasures marked, then asking them to find treasure on a blank map.

---

### Problem 3: Distribution Mismatch

**What Transformer Learned:**

During training, the Transformer saw histories like:
```
History at step t=5: [5 points from 10k pool]
  - Already diverse (random sampling covered space)
  - GP trained on 10k points = confident
  - Teacher picks from remaining 9,995 evaluated points
  - Next point: one of these 9,995 (high cEI)
```

**What Transformer Faces at Test Time:**

At step t=5:
```
History: [5 points from online sequential selection]
  - Not diverse (sequentially chosen, might cluster)
  - No GP available (Transformer doesn't use GPs)
  - Must suggest point in [0,1]^D (infinite options!)
  - Next point: anywhere (no pool to choose from)
```

**The Problem:**
- Training: "Pick from a pool of good options"
- Testing: "Generate a good option from scratch"

These are DIFFERENT TASKS!

**Analogy:** Training a chef by having them select dishes from a menu, then asking them to cook from scratch with raw ingredients.

---

## Contrast with qLogEI (Always Online)

### qLogEI Workflow:

```
Step 1: Initial design (4 random points)
  Evaluate x₁, x₂, x₃, x₄
  Budget: 4/20

Step 2: Build GP from these 4 points
  GP_f = fit_gp([x₁,x₂,x₃,x₄], [y₁,y₂,y₃,y₄])
  GP_c = fit_gp([x₁,x₂,x₃,x₄], [c₁,c₂,c₃,c₄])

Step 3: Compute qLogEI over candidate points
  Sample 128 candidates: x'₁, x'₂, ..., x'₁₂₈
  For each: compute qLogEI(x'ᵢ | GP_f, GP_c, history)
  Pick: x₅ = argmax qLogEI

Step 4: Evaluate x₅
  Get y₅, c₅
  Budget: 5/20

Step 5: Update GP with new point
  GP_f = fit_gp([x₁,x₂,x₃,x₄,x₅], [y₁,y₂,y₃,y₄,y₅])
  GP_c = fit_gp([x₁,x₂,x₃,x₄,x₅], [c₁,c₂,c₃,c₄,c₅])

Step 6: Compute qLogEI again (with updated GP)
  Pick x₆ = argmax qLogEI
  ...

Continue until budget exhausted
```

**Key Differences:**

| Aspect | Transformer | qLogEI |
|--------|-------------|--------|
| **Training** | Offline (10k points) | None (no training) |
| **GP usage** | Teacher used GPs offline | Builds GP online from test data |
| **Adaptation** | Fixed policy | Adapts GP at every step |
| **Uncertainty** | None | Full GP posterior μ(x), σ(x) |
| **Exploration** | Learned pattern (doesn't transfer) | Principled (high σ → explore) |

**Why qLogEI Wins:**
- No offline-online mismatch (always online!)
- GP trained on ACTUAL test problem
- Adapts as it learns
- Proper uncertainty quantification

---

## Visualizing the Problem

### Information Available at Step t=10:

```
TRAINING (Offline):
┌─────────────────────────────────────────────────────┐
│  Search Space [0,1]²                                │
│                                                     │
│  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │
│  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │
│  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │
│  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │
│  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │
│  ● = evaluated (10,000 points!)                     │
│  ✓ = history (10 points selected from pool)         │
│                                                     │
│  Teacher sees EVERYTHING, picks best next point    │
└─────────────────────────────────────────────────────┘

TESTING (Online):
┌─────────────────────────────────────────────────────┐
│  Search Space [0,1]²                                │
│                                                     │
│  ✓                                                  │
│              ✓        ✓                             │
│                                                     │
│    ✓                              ✓                 │
│                 ✓                                   │
│  ✓         ✓                  ✓                     │
│                        ✓                            │
│  ✓ = evaluated (only 10 points!)                   │
│                                                     │
│  Transformer sees 10 points, must guess where's good│
└─────────────────────────────────────────────────────┘
```

---

## The Fundamental Mismatch

### What Transformer Learned:

> "Given a history of points from a dense offline dataset, predict which point from the remaining pool the teacher (cEI) would select."

### What Transformer Must Do:

> "Given a history of points from sparse online sampling, generate a new point in continuous space that will yield good f(x) and feasible g(x)."

**These are COMPLETELY DIFFERENT problems!**

---

## Why This Causes Poor Performance

### Example Failure Mode:

**Training Scenario:**
```
Offline pool includes:
  x₁₂₃ = [0.7, 0.3] → f=200, g=-0.5 (feasible, GOOD!)
  x₄₅₆ = [0.75, 0.35] → f=180, g=-0.3 (feasible, BETTER!)
  x₇₈₉ = [0.72, 0.32] → f=190, g=-0.4 (feasible, good)

Teacher picks x₄₅₆ (best from pool)

Transformer learns:
  "When history shows points near [0.7, 0.3], pick x₄₅₆"
```

**Testing Scenario:**
```
History has:
  x₁ = [0.69, 0.31] → f=1200, g=0.2 (infeasible)
  x₂ = [0.71, 0.29] → f=1500, g=0.5 (infeasible)

Transformer pattern-matches:
  "History near [0.7, 0.3], predict x₄₅₆ = [0.75, 0.35]"

Reality:
  Evaluate x₄₅₆ → f=2100, g=0.8 (infeasible, BAD!)

Why? Because this is a DIFFERENT PROBLEM (F6 vs F2)!
The pattern doesn't transfer!
```

**qLogEI Instead:**
```
History:
  x₁ = [0.69, 0.31] → f=1200, g=0.2 (infeasible)
  x₂ = [0.71, 0.29] → f=1500, g=0.5 (infeasible)

Build GP from x₁, x₂:
  μ_f([0.7, 0.3]) ≈ 1350 (interpolate)
  σ_f([0.7, 0.3]) ≈ high (only 2 points nearby)
  μ_c([0.7, 0.3]) ≈ 0.35 (interpolate)
  σ_c([0.7, 0.3]) ≈ high (uncertain)

Compute EI:
  Region [0.7, 0.3]: high uncertainty → explore elsewhere!

Pick x₃ = [0.2, 0.8] (different region, high exploration value)
Evaluate → f=450, g=-0.1 (feasible!) ✓

qLogEI adapts to THIS problem, not patterns from training!
```

---

## Key Takeaway

**Offline vs Online Mismatch:**

> "The Transformer was trained in an offline setting with complete information (10k pre-evaluated points), but tested in an online setting with sparse, sequential information (20 budget). This is like training a pilot in a simulator with full visibility, then asking them to fly blind in a storm. qLogEI succeeds because it adapts online—building its model (GP) from actual test data rather than relying on patterns learned from different problems offline."

---

## For Your Poster

**Simple Explanation:**

> **"Distribution Shift (Offline → Online)"**
>
> Training: Teacher (cEI) selects from 10,000 pre-evaluated points—exploration already done, just exploitation.
>
> Testing: Transformer must explore AND exploit with only 20 evaluations—a completely different problem.
>
> Result: Learned patterns don't transfer. The Transformer never learned to explore because it never needed to during training.
>
> qLogEI succeeds because it builds its GP model online from actual test data, adapting to each specific problem rather than relying on offline patterns.

**Diagram for Poster:**

```
TRAINING (Offline)               TESTING (Online)
┌────────────────────┐          ┌────────────────────┐
│ 10k points ●●●●●● │          │ 4 points ✓         │
│ known ●●●●●●●●●●●● │    →     │ 16 unknown ?????   │
│ Teacher: pick best │          │ Must explore!      │
└────────────────────┘          └────────────────────┘
     Dense info                      Sparse info
     Exploitation                    Explore + Exploit
     Different problem!
```

Does this clarify the offline vs online distinction and why it matters?
