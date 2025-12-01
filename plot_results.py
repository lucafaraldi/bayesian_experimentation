#!/usr/bin/env python
"""
Generate convergence plots and summary tables for the poster.

Usage:
    python plot_results.py

Requires:
    - results_coco_rl_vs_qlogei.csv (from benchmark_rl.py)
    - results_coco_rl_vs_qlogei_summary.csv (from benchmark_rl.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
RESULTS_CSV = Path("results_coco_rl_vs_qlogei.csv")
SUMMARY_CSV = Path("results_coco_rl_vs_qlogei_summary.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors for methods
METHOD_COLORS = {
    "Random": "#e74c3c",      # Red
    "qLogEI": "#3498db",      # Blue
    "RL": "#2ecc71",          # Green
    "Transformer": "#f39c12", # Orange
}

METHOD_LABELS = {
    "Random": "Random Search",
    "qLogEI": "qLogEI (BoTorch)",
    "RL": "RL Policy (PPO)",
    "Transformer": "Transformer",
}


def load_data():
    """Load benchmark results."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Run benchmark_rl.py first!"
        )

    df = pd.read_csv(RESULTS_CSV)

    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
    else:
        summary_df = None

    return df, summary_df


def plot_convergence_by_dimension(df):
    """
    Create convergence plots: one per dimension.
    Averaged over all functions, instances, and repetitions.
    """
    dims = sorted(df['dim'].unique())
    methods = sorted(df['method'].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        for method in methods:
            subset = df[(df['dim'] == dim) & (df['method'] == method)]

            # Group by evaluation number and compute mean ± std across all runs
            grouped = subset.groupby('eval')['best_feasible'].agg(['mean', 'std'])
            evals = grouped.index.values
            means = grouped['mean'].values
            stds = grouped['std'].values

            color = METHOD_COLORS.get(method, "#95a5a6")
            label = METHOD_LABELS.get(method, method)

            ax.plot(evals, means, label=label, color=color, linewidth=2)
            ax.fill_between(
                evals,
                means - stds,
                means + stds,
                alpha=0.2,
                color=color
            )

        ax.set_xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Feasible Objective', fontsize=12, fontweight='bold')
        ax.set_title(f'Dimension {dim}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Log scale if values span multiple orders of magnitude
        y_range = means.max() - means.min()
        if y_range > 100:
            ax.set_yscale('log')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "convergence_by_dimension.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_convergence_by_function(df, representative_functions=None):
    """
    Create convergence plots for representative functions.
    One plot per function, showing both dimensions.
    """
    if representative_functions is None:
        # Pick 3 representative functions
        representative_functions = [2, 6, 50]

    available_funcs = sorted(df['function'].unique())
    funcs_to_plot = [f for f in representative_functions if f in available_funcs]

    if not funcs_to_plot:
        print("No representative functions available in data.")
        return

    dims = sorted(df['dim'].unique())
    methods = sorted(df['method'].unique())

    fig, axes = plt.subplots(
        len(dims), len(funcs_to_plot),
        figsize=(5 * len(funcs_to_plot), 4 * len(dims)),
        squeeze=False
    )

    for row, dim in enumerate(dims):
        for col, func in enumerate(funcs_to_plot):
            ax = axes[row, col]

            for method in methods:
                subset = df[
                    (df['dim'] == dim) &
                    (df['function'] == func) &
                    (df['method'] == method)
                ]

                if subset.empty:
                    continue

                # Average over instances and repetitions
                grouped = subset.groupby('eval')['best_feasible'].agg(['mean', 'std'])
                evals = grouped.index.values
                means = grouped['mean'].values
                stds = grouped['std'].values

                color = METHOD_COLORS.get(method, "#95a5a6")
                label = METHOD_LABELS.get(method, method)

                ax.plot(evals, means, label=label, color=color, linewidth=2)
                ax.fill_between(
                    evals,
                    means - stds,
                    means + stds,
                    alpha=0.2,
                    color=color
                )

            ax.set_title(f'F{func}, D={dim}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Evaluations', fontsize=10)
            ax.set_ylabel('Best Feasible', fontsize=10)
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 0:
                ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "convergence_by_function.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(summary_df):
    """
    Create a summary table of final best feasible values.
    """
    if summary_df is None:
        print("Summary CSV not found, skipping table.")
        return

    # Group by method and dimension
    summary = summary_df.groupby(['method', 'dim'])['final_best_feasible'].agg(
        ['mean', 'std', 'count']
    ).reset_index()

    # Pivot to wide format
    pivot = summary.pivot(index='method', columns='dim', values=['mean', 'std'])

    print("\n" + "="*80)
    print("SUMMARY TABLE: Final Best Feasible Values")
    print("="*80)
    print(pivot.to_string())
    print("="*80 + "\n")

    # Also save to CSV
    output_path = OUTPUT_DIR / "summary_table.csv"
    pivot.to_csv(output_path)
    print(f"Saved: {output_path}")

    # Rank methods by overall performance (across all dims)
    overall = summary_df.groupby('method')['final_best_feasible'].mean().sort_values()
    print("\nOVERALL RANKING (lower is better):")
    for rank, (method, value) in enumerate(overall.items(), start=1):
        print(f"  {rank}. {METHOD_LABELS.get(method, method)}: {value:.3e}")
    print()


def plot_performance_heatmap(summary_df):
    """
    Create a heatmap showing performance by function and method.
    """
    if summary_df is None:
        print("Summary CSV not found, skipping heatmap.")
        return

    # Average over instances and repetitions
    perf = summary_df.groupby(['method', 'function', 'dim'])['final_best_feasible'].mean().reset_index()

    dims = sorted(perf['dim'].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        subset = perf[perf['dim'] == dim]
        pivot = subset.pivot(index='method', columns='function', values='final_best_feasible')

        # Reorder methods for consistent display
        method_order = [m for m in ['Random', 'qLogEI', 'RL', 'Transformer'] if m in pivot.index]
        pivot = pivot.reindex(method_order)

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2e',
            cmap='RdYlGn_r',  # Red=bad, Green=good (reversed for minimization)
            cbar_kws={'label': 'Best Feasible'},
            ax=ax,
            linewidths=0.5
        )
        ax.set_title(f'Performance Heatmap (Dim={dim})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Function ID', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "performance_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_final_performance_bars(summary_df):
    """
    Bar chart comparing final performance of all methods.
    """
    if summary_df is None:
        print("Summary CSV not found, skipping bar chart.")
        return

    # Average over all problems
    overall = summary_df.groupby(['method', 'dim'])['final_best_feasible'].mean().reset_index()

    dims = sorted(overall['dim'].unique())
    methods = sorted(overall['method'].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        subset = overall[overall['dim'] == dim]

        # Sort by performance
        subset = subset.sort_values('final_best_feasible', ascending=True)

        colors = [METHOD_COLORS.get(m, "#95a5a6") for m in subset['method']]
        labels = [METHOD_LABELS.get(m, m) for m in subset['method']]

        bars = ax.barh(labels, subset['final_best_feasible'], color=colors)
        ax.set_xlabel('Final Best Feasible (lower is better)', fontsize=11, fontweight='bold')
        ax.set_title(f'Dimension {dim}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, subset['final_best_feasible'])):
            ax.text(
                val, i,
                f' {val:.2e}',
                va='center',
                fontsize=9
            )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "final_performance_bars.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("GENERATING PLOTS FOR POSTER")
    print("="*80 + "\n")

    # Load data
    df, summary_df = load_data()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    # Print basic stats
    print(f"\nMethods: {sorted(df['method'].unique())}")
    print(f"Dimensions: {sorted(df['dim'].unique())}")
    print(f"Functions: {sorted(df['function'].unique())}")
    print(f"Instances: {sorted(df['instance'].unique())}")
    print(f"Repetitions: {sorted(df['repetition'].unique())}\n")

    # Generate plots
    print("Generating convergence plot by dimension...")
    plot_convergence_by_dimension(df)

    print("Generating convergence plots for representative functions...")
    plot_convergence_by_function(df, representative_functions=[2, 6, 50])

    if summary_df is not None:
        print("Creating summary table...")
        create_summary_table(summary_df)

        print("Generating performance heatmap...")
        plot_performance_heatmap(summary_df)

        print("Generating final performance bar chart...")
        plot_final_performance_bars(summary_df)

    print("\n" + "="*80)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
