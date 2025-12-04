"""
Generate Comparison Figures from Saved Results

This script loads saved comparison results from JSON and generates
4 comparison plots with the figure map positioned on the left.

Responsibilities:
- Load results and statistics from comparison_results.json
- Generate 4 comparison plots (best colors, average colors, execution time, conflicts)
- Position figure map on the left side of each plot
- Save figures to a new folder

Dependencies:
- pandas, numpy, matplotlib for data processing and plotting
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_results(json_file):
    """
    Load results from comparison_results.json.
    
    Args:
        json_file: Path to comparison_results.json
        
    Returns:
        Tuple of (df_results, df_stats, best_known)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df_results = pd.DataFrame(data['results'])
    df_stats = pd.DataFrame(data['statistics'])
    best_known = data.get('best_known', {})
    
    return df_results, df_stats, best_known


def generate_figures(df_results, df_stats, output_dir, best_known=None):
    """
    Generate 4 comparison plots with figure map on the left.
    
    Args:
        df_results: DataFrame with raw results
        df_stats: DataFrame with statistics
        output_dir: Directory to save plots
        best_known: Optional dictionary mapping graph names to best known color counts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    graphs = df_stats['graph'].unique()
    algorithms = df_stats['algorithm'].unique()
    has_best_known = best_known is not None and len(best_known) > 0
    
    # Consistent colors for algorithms across all plots
    ALGO_COLORS = {
        'Greedy': '#1f77b4',
        'Tabu Search': '#ff7f0e',
        'ACO': '#2ca02c'
    }
    
    x = np.arange(len(graphs))
    n_algos = len(algorithms)
    
    print("\nGenerating comparison figures...")
    print("=" * 80)
    
    # Plot 1: Best Color Count (with Best Known Solution as bars)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / (n_algos + (1 if has_best_known else 0))
    
    # Plot BKS first as separate bars if available
    bar_index = 0
    if has_best_known:
        bks_values = [best_known.get(g, 0) for g in graphs]
        bars = ax.bar(x + bar_index * bar_width, bks_values, bar_width, 
               label='Best Known Solution', color='gold', edgecolor='black', linewidth=1.5)
        # Add text labels inside bars
        for i, (bar, val) in enumerate(zip(bars, bks_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{int(val)}', ha='center', va='center', fontweight='bold', fontsize=9)
        bar_index += 1
    
    # Plot algorithm results - bars touching
    for algo in algorithms:
        algo_data = df_stats[df_stats['algorithm'] == algo].sort_values('graph')
        best_values = [algo_data[algo_data['graph'] == g]['best_colors'].values[0] for g in graphs]
        bars = ax.bar(x + bar_index * bar_width, best_values, bar_width, 
               label=algo, color=ALGO_COLORS.get(algo, 'gray'), edgecolor='black', linewidth=0.5)
        # Add text labels inside bars
        for i, (bar, val) in enumerate(zip(bars, best_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{int(val)}', ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        bar_index += 1
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Best Color Count', fontweight='bold', fontsize=12)
    ax.set_title('Best Color Count Comparison (Algorithms vs Best Known Solution)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    best_colors_file = output_path / 'comparison_best_colors.png'
    fig.savefig(best_colors_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1/4 created: Best Color Count Comparison")
    
    # Plot 2: Average Color Count with Std Dev (with BKS as bars)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_index = 0
    # Plot BKS if available
    if has_best_known:
        bks_values = [best_known.get(g, 0) for g in graphs]
        bars = ax.bar(x + bar_index * bar_width, bks_values, bar_width, 
               label='Best Known Solution', color='gold', edgecolor='black', linewidth=1.5)
        # Add text labels inside bars
        for i, (bar, val) in enumerate(zip(bars, bks_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{int(val)}', ha='center', va='center', fontweight='bold', fontsize=9)
        bar_index += 1
    
    # Plot algorithm averages with error bars - bars touching
    for algo in algorithms:
        algo_data = df_stats[df_stats['algorithm'] == algo].sort_values('graph')
        avg_values = [algo_data[algo_data['graph'] == g]['avg_colors'].values[0] for g in graphs]
        std_values = [algo_data[algo_data['graph'] == g]['std_colors'].values[0] for g in graphs]
        bars = ax.bar(x + bar_index * bar_width, avg_values, bar_width, 
               yerr=std_values, capsize=5, label=algo, 
               color=ALGO_COLORS.get(algo, 'gray'), edgecolor='black', linewidth=0.5)
        # Add text labels inside bars
        for i, (bar, val) in enumerate(zip(bars, avg_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{val:.1f}', ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        bar_index += 1
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Color Count', fontweight='bold', fontsize=12)
    ax.set_title('Average Color Count Comparison (with Standard Deviation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    avg_colors_file = output_path / 'comparison_avg_colors.png'
    fig.savefig(avg_colors_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2/4 created: Average Color Count Comparison")
    
    # Plot 3: Average Execution Time
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_width = 0.25
    for i, algo in enumerate(algorithms):
        algo_data = df_stats[df_stats['algorithm'] == algo].sort_values('graph')
        time_values = [algo_data[algo_data['graph'] == g]['avg_time'].values[0] for g in graphs]
        bars = ax.bar(x + i * algo_width, time_values, algo_width, 
               label=algo, color=ALGO_COLORS.get(algo, 'gray'), edgecolor='black', linewidth=0.5)
        # Add text labels inside bars
        for j, (bar, val) in enumerate(zip(bars, time_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*0.80, 
                   f'{val:.1f}', ha='center', va='center', fontweight='bold', fontsize=8, color='black')
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Execution Time (s)', fontweight='bold', fontsize=12)
    ax.set_title('Execution Time Comparison (log scale)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + algo_width * (n_algos - 1) / 2)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, which='both')
    ax.set_yscale('log')
    plt.tight_layout()
    
    exec_time_file = output_path / 'comparison_execution_time.png'
    fig.savefig(exec_time_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3/4 created: Execution Time Comparison")
    
    # Plot 4: Average Conflict Count
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        algo_data = df_stats[df_stats['algorithm'] == algo].sort_values('graph')
        conflict_values = [algo_data[algo_data['graph'] == g]['avg_conflicts'].values[0] for g in graphs]
        bars = ax.bar(x + i * algo_width, conflict_values, algo_width, 
               label=algo, color=ALGO_COLORS.get(algo, 'gray'), edgecolor='black', linewidth=0.5)
        # Add text labels inside bars (only if non-zero)
        for j, (bar, val) in enumerate(zip(bars, conflict_values)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                       f'{val:.1f}', ha='center', va='center', fontweight='bold', fontsize=9, color='white')
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Conflict Count', fontweight='bold', fontsize=12)
    ax.set_title('Conflict Count Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + algo_width * (n_algos - 1) / 2)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    conflicts_file = output_path / 'comparison_conflicts.png'
    fig.savefig(conflicts_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4/4 created: Conflict Count Comparison")
    
    print("=" * 80)
    print(f"✓ All figures saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate comparison figures from saved results'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to comparison_results.json file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures (default: figures/ in same directory as JSON)'
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Default output directory is figures/ in same directory as JSON
    if args.output_dir is None:
        output_dir = json_path.parent / 'figures'
    else:
        output_dir = Path(args.output_dir)
    
    print(f"Loading results from: {json_path}")
    df_results, df_stats, best_known = load_results(json_path)
    
    print(f"Output directory: {output_dir}")
    generate_figures(df_results, df_stats, output_dir, best_known)


if __name__ == '__main__':
    main()
