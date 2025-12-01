#!/usr/bin/env python
"""
Compare performance across all three RL versions.

Usage:
    python compare_rl_versions.py

Requirements:
    - results_coco_rl_vs_qlogei_summary.csv (Original RL)
    - results_coco_rl_enhanced_vs_qlogei_summary.csv (Enhanced RL)
    - results_coco_rl_full_vs_qlogei_summary.csv (Full COCO RL)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# File paths
ORIG_CSV = Path("results_coco_rl_vs_qlogei_summary.csv")
ENH_CSV = Path("results_coco_rl_enhanced_vs_qlogei_summary.csv")
FULL_CSV = Path("results_coco_rl_full_vs_qlogei_summary.csv")

def load_results():
    """Load all three result files."""
    results = {}

    if ORIG_CSV.exists():
        df = pd.read_csv(ORIG_CSV)
        results['Original'] = df[df['method'] == 'RL'].copy()
        results['Original']['version'] = 'Original'
        print(f"✓ Loaded {len(results['Original'])} results from Original RL")
    else:
        print(f"⚠ Missing: {ORIG_CSV}")

    if ENH_CSV.exists():
        df = pd.read_csv(ENH_CSV)
        results['Enhanced'] = df[df['method'] == 'RL_Enhanced'].copy()
        results['Enhanced']['version'] = 'Enhanced'
        print(f"✓ Loaded {len(results['Enhanced'])} results from Enhanced RL")
    else:
        print(f"⚠ Missing: {ENH_CSV}")

    if FULL_CSV.exists():
        df = pd.read_csv(FULL_CSV)
        results['Full'] = df[df['method'] == 'RL_Full'].copy()
        results['Full']['version'] = 'Full_COCO'
        print(f"✓ Loaded {len(results['Full'])} results from Full COCO RL")
    else:
        print(f"⚠ Missing: {FULL_CSV}")

    if len(results) < 2:
        raise ValueError("Need at least 2 result files to compare!")

    return results

def overall_statistics(results):
    """Compute overall statistics for each version."""
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    stats_rows = []
    for version, df in results.items():
        scores = df['final_best_feasible'].values

        # Remove inf values for stats
        finite_scores = scores[np.isfinite(scores)]

        stats_rows.append({
            'Version': version,
            'Mean': np.mean(finite_scores) if len(finite_scores) > 0 else np.inf,
            'Median': np.median(finite_scores) if len(finite_scores) > 0 else np.inf,
            'Std': np.std(finite_scores) if len(finite_scores) > 0 else np.inf,
            'Min': np.min(finite_scores) if len(finite_scores) > 0 else np.inf,
            'Max': np.max(finite_scores) if len(finite_scores) > 0 else np.inf,
            'N_finite': len(finite_scores),
            'N_inf': np.sum(np.isinf(scores)),
        })

    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    return stats_df

def win_rate_analysis(results):
    """Compute win rates (lower is better)."""
    print("\n" + "="*70)
    print("WIN RATE ANALYSIS (% of problems where each version wins)")
    print("="*70)

    # Combine all results
    all_df = pd.concat(results.values(), ignore_index=True)

    # Pivot to get side-by-side comparison
    pivot = all_df.pivot_table(
        values='final_best_feasible',
        index=['function', 'instance', 'dim', 'repetition'],
        columns='version'
    )

    # Count wins (handle missing columns gracefully)
    versions = list(results.keys())
    win_counts = {}

    for version in versions:
        if version not in pivot.columns:
            continue

        # This version wins if its score is lower than all others
        other_versions = [v for v in versions if v != version and v in pivot.columns]
        if not other_versions:
            continue

        is_winner = (pivot[version] < pivot[other_versions].min(axis=1))
        win_counts[version] = is_winner.sum()

    total_problems = len(pivot)

    print(f"\nTotal problems: {total_problems}")
    print(f"\nWin counts (lower best_feasible wins):")
    for version, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_problems
        print(f"  {version:15s}: {count:3d} / {total_problems} ({pct:.1f}%)")

    return win_counts

def per_function_analysis(results):
    """Analyze performance per function."""
    print("\n" + "="*70)
    print("PER-FUNCTION ANALYSIS")
    print("="*70)

    all_df = pd.concat(results.values(), ignore_index=True)

    # Group by function and version
    grouped = all_df.groupby(['function', 'version'])['final_best_feasible'].agg([
        ('mean', lambda x: np.mean(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.inf),
        ('median', lambda x: np.median(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.inf),
        ('n_finite', lambda x: np.sum(np.isfinite(x))),
    ]).reset_index()

    # Pivot for easier comparison
    pivot_mean = grouped.pivot(index='function', columns='version', values='mean')

    print("\nMean final best feasible per function:")
    print(pivot_mean.to_string())

    return pivot_mean

def pairwise_statistical_tests(results):
    """Perform pairwise statistical tests."""
    print("\n" + "="*70)
    print("PAIRWISE STATISTICAL TESTS (paired t-test)")
    print("="*70)

    versions = list(results.keys())

    if len(versions) < 2:
        print("Need at least 2 versions for comparison")
        return

    # Combine and pivot
    all_df = pd.concat(results.values(), ignore_index=True)
    pivot = all_df.pivot_table(
        values='final_best_feasible',
        index=['function', 'instance', 'dim', 'repetition'],
        columns='version'
    )

    # Pairwise tests
    for i, v1 in enumerate(versions):
        for v2 in versions[i+1:]:
            if v1 not in pivot.columns or v2 not in pivot.columns:
                continue

            # Get paired samples (both finite)
            mask = np.isfinite(pivot[v1]) & np.isfinite(pivot[v2])
            scores1 = pivot.loc[mask, v1].values
            scores2 = pivot.loc[mask, v2].values

            if len(scores1) < 3:
                print(f"\n{v1} vs {v2}: Not enough paired samples ({len(scores1)})")
                continue

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)

            # Effect size (Cohen's d)
            mean_diff = np.mean(scores1 - scores2)
            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

            # Determine winner
            if mean_diff < 0:
                winner = v1
            elif mean_diff > 0:
                winner = v2
            else:
                winner = "tie"

            print(f"\n{v1} vs {v2}:")
            print(f"  Paired samples: {len(scores1)}")
            print(f"  Mean {v1}: {np.mean(scores1):.3e}")
            print(f"  Mean {v2}: {np.mean(scores2):.3e}")
            print(f"  Mean difference: {mean_diff:.3e} (positive = {v2} better)")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}")

            if p_value < 0.05:
                print(f"  ✓ Statistically significant (p < 0.05)")
                print(f"  Winner: {winner}")
            else:
                print(f"  ✗ Not statistically significant (p >= 0.05)")

def per_dimension_analysis(results):
    """Analyze performance per dimension."""
    print("\n" + "="*70)
    print("PER-DIMENSION ANALYSIS")
    print("="*70)

    all_df = pd.concat(results.values(), ignore_index=True)

    for dim in sorted(all_df['dim'].unique()):
        print(f"\n--- Dimension = {dim} ---")
        dim_df = all_df[all_df['dim'] == dim]

        for version in sorted(dim_df['version'].unique()):
            version_df = dim_df[dim_df['version'] == version]
            scores = version_df['final_best_feasible'].values
            finite_scores = scores[np.isfinite(scores)]

            if len(finite_scores) > 0:
                mean_score = np.mean(finite_scores)
                median_score = np.median(finite_scores)
                n_inf = np.sum(np.isinf(scores))
                print(f"  {version:15s}: mean={mean_score:10.3e}, median={median_score:10.3e}, n_inf={n_inf}")
            else:
                print(f"  {version:15s}: All inf")

def main():
    print("="*70)
    print("RL VERSION COMPARISON")
    print("="*70)

    # Load results
    results = load_results()

    # Run analyses
    overall_statistics(results)
    win_rate_analysis(results)
    per_function_analysis(results)
    per_dimension_analysis(results)
    pairwise_statistical_tests(results)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
