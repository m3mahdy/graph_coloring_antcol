"""
Comprehensive Testing Script for Graph Coloring Algorithms

This script compares three graph coloring algorithms (Greedy, Tabu Search, and ACO)
on a test dataset. It runs multiple repetitions, calculates statistics, and generates
comparison visualizations.

Responsibilities:
- Run all 3 algorithms on test graphs with multiple repetitions
- Calculate statistics (best, avg, std) for each algorithm
- Generate per-algorithm plots via testing_utils
- Generate comparison plots across all algorithms
- Cache results in JSON format

Dependencies:
- dataloader: Load graphs and best known results
- testing_utils: Generate per-algorithm plots
- Algorithm modules: greedy_algorithm, tabu_search_algorithm, aco_gpc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

from dataloader import GraphDataLoader
from testing_utils import generate_algorithm_plots
from greedy_algorithm import GreedyGraphColoring
from tabu_search_algorithm import TabuSearchGraphColoring
from aco_gpc import ACOGraphColoring


ALGORITHM_NAMES = {
    'greedy': 'Greedy',
    'tabu': 'Tabu Search',
    'aco': 'ACO'
}


def run_algorithm_single(algorithm_name, graph, params=None):
    """
    Run a single algorithm on a graph once.
    
    Args:
        algorithm_name: Name of the algorithm ('greedy', 'tabu', 'aco')
        graph: NetworkX graph
        params: Optional parameters dictionary
        
    Returns:
        Dictionary with result metrics
    """
    if params is None:
        params = {}
    
    start_time = time.time()
    
    if algorithm_name == 'greedy':
        algo = GreedyGraphColoring(graph, verbose=False)
        result = algo.run()
    elif algorithm_name == 'tabu':
        max_iter = params.get('max_iterations', 7000)
        tabu_size = params.get('tabu_size', 10)
        tabu_reps = params.get('tabu_reps', 50)
        algo = TabuSearchGraphColoring(
            graph,
            max_iterations=max_iter,
            tabu_size=tabu_size,
            tabu_reps=tabu_reps,
            verbose=False
        )
        result = algo.run()
    elif algorithm_name == 'aco':
        iterations = params.get('iterations', 30)
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 2.0)
        rho = params.get('rho', 0.1)
        ant_count = params.get('ant_count', 10)
        Q = params.get('Q', 1.0)
        algo = ACOGraphColoring(
            graph,
            iterations=iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            ant_count=ant_count,
            Q=Q,
            verbose=False
        )
        result = algo.run()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    elapsed_time = time.time() - start_time
    
    # Handle different return formats: Greedy/Tabu return conflict_count, ACO does not
    conflict_count = result.get('conflict_count', 0)  # ACO guarantees 0 conflicts
    
    return {
        'color_count': result['color_count'],
        'conflict_count': conflict_count,
        'execution_time': elapsed_time
    }


def run_comprehensive_test(test_graphs, output_dir, num_repetitions=5, aco_params=None):
    """
    Run comprehensive testing comparing all algorithms.
    
    Args:
        test_graphs: List of tuples (filename, graph) from dataloader
        output_dir: Directory to save results and plots
        num_repetitions: Number of times to run each algorithm (default: 5)
        aco_params: Optional parameters for ACO algorithm
        
    Returns:
        DataFrame with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not test_graphs:
        raise ValueError("No test graphs provided")
    
    print(f"Found {len(test_graphs)} test graphs")
    print(f"Running {num_repetitions} repetitions per graph per algorithm")
    print("=" * 80)
    
    all_results = []
    
    for graph_name, graph in test_graphs:
        print(f"\nProcessing: {graph_name}")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        for algorithm in ['greedy', 'tabu', 'aco']:
            print(f"  Running {ALGORITHM_NAMES[algorithm]}...")
            
            params = aco_params if algorithm == 'aco' and aco_params else {}
            
            for rep in range(num_repetitions):
                try:
                    result = run_algorithm_single(algorithm, graph, params)
                    
                    all_results.append({
                        'graph': graph_name,
                        'algorithm': algorithm,
                        'repetition': rep + 1,
                        'color_count': result['color_count'],
                        'conflict_count': result['conflict_count'],
                        'execution_time': result['execution_time']
                    })
                    
                    print(f"    Rep {rep + 1}: colors={result['color_count']}, "
                          f"conflicts={result['conflict_count']}, "
                          f"time={result['execution_time']:.3f}s")
                
                except Exception as e:
                    print(f"    Rep {rep + 1}: ERROR - {e}")
    
    df_results = pd.DataFrame(all_results)
    
    results_file = output_path / 'comprehensive_test_results.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to CSV (comprehensive_test_results.csv)")
    
    return df_results


def calculate_statistics(df_results, best_known=None):
    """
    Calculate statistics from test results.
    
    Args:
        df_results: DataFrame with test results
        best_known: Optional dictionary mapping graph names to best known color counts
        
    Returns:
        DataFrame with statistics
    """
    if best_known is None:
        best_known = {}
    
    stats_list = []
    
    for graph_name in df_results['graph'].unique():
        df_graph = df_results[df_results['graph'] == graph_name]
        bks = best_known.get(graph_name, None)
        
        for algorithm in df_graph['algorithm'].unique():
            df_algo = df_graph[df_graph['algorithm'] == algorithm]
            
            best_colors = int(df_algo['color_count'].min())
            avg_colors = float(df_algo['color_count'].mean())
            
            stat_entry = {
                'graph': graph_name,
                'algorithm': ALGORITHM_NAMES[algorithm],
                'best_colors': best_colors,
                'avg_colors': avg_colors,
                'std_colors': float(df_algo['color_count'].std()),
                'avg_conflicts': float(df_algo['conflict_count'].mean()),
                'avg_time': float(df_algo['execution_time'].mean()),
                'std_time': float(df_algo['execution_time'].std())
            }
            
            # Add best known solution and deviation if available
            if bks is not None:
                stat_entry['best_known'] = bks
                stat_entry['deviation_best'] = float((best_colors - bks) / bks * 100)
                stat_entry['deviation_avg'] = float((avg_colors - bks) / bks * 100)
            
            stats_list.append(stat_entry)
    
    df_stats = pd.DataFrame(stats_list)
    return df_stats


def generate_comparison_plots(df_results, df_stats, output_dir, best_known=None):
    """
    Generate 4 comparison plots for the test results.
    
    Args:
        df_results: DataFrame with raw results
        df_stats: DataFrame with statistics
        output_dir: Directory to save plots
        best_known: Optional dictionary mapping graph names to best known color counts
    """
    output_path = Path(output_dir)
    
    graphs = df_stats['graph'].unique()
    algorithms = df_stats['algorithm'].unique()
    has_best_known = best_known is not None and len(best_known) > 0
    
    # Consistent colors for algorithms across all plots
    ALGO_COLORS = {
        'Greedy': '#1f77b4',      # Blue
        'Tabu Search': '#ff7f0e', # Orange
        'ACO': '#2ca02c'          # Green
    }
    
    x = np.arange(len(graphs))
    n_algos = len(algorithms)
    
    # Plot 1: Best Color Count (with Best Known Solution as bars)
    # Bars touching each other - no space between algorithm bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / (n_algos + (1 if has_best_known else 0))  # Total width = 0.8, divided by all bars
    
    # Plot BKS first as separate bars if available
    bar_index = 0
    if has_best_known:
        bks_values = [best_known.get(g, np.nan) for g in graphs]
        # Filter out NaN values for plotting
        bks_positions = []
        bks_heights = []
        for i, v in enumerate(bks_values):
            if not np.isnan(v):
                bks_positions.append(i + bar_index * bar_width)
                bks_heights.append(v)
        ax.bar(bks_positions, bks_heights, bar_width, label='Best Known Solution', 
               color='gold', edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)
        bar_index += 1
    
    # Plot algorithm results - bars touching
    for algo in algorithms:
        df_algo = df_stats[df_stats['algorithm'] == algo]
        best_colors = [df_algo[df_algo['graph'] == g]['best_colors'].values[0] if len(df_algo[df_algo['graph'] == g]) > 0 else 0 for g in graphs]
        ax.bar(x + bar_index * bar_width, best_colors, bar_width, label=algo, 
               color=ALGO_COLORS.get(algo, '#888888'), alpha=0.85)
        bar_index += 1
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Best Color Count', fontweight='bold', fontsize=12)
    ax.set_title('Best Color Count Comparison (Algorithms vs Best Known Solution)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 0.4)  # Center ticks
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    best_colors_file = output_path / 'comparison_best_colors.png'
    if best_colors_file.exists():
        best_colors_file.unlink()
    fig.savefig(best_colors_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1/4 created: Best Color Count Comparison")
    
    # Plot 2: Average Color Count with Std Dev (with BKS as bars)
    # Bars touching each other
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_index = 0
    # Plot BKS if available
    if has_best_known:
        bks_values = [best_known.get(g, np.nan) for g in graphs]
        bks_positions = []
        bks_heights = []
        for i, v in enumerate(bks_values):
            if not np.isnan(v):
                bks_positions.append(i + bar_index * bar_width)
                bks_heights.append(v)
        ax.bar(bks_positions, bks_heights, bar_width, label='Best Known Solution', 
               color='gold', edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)
        bar_index += 1
    
    # Plot algorithm averages with error bars - bars touching
    for algo in algorithms:
        df_algo = df_stats[df_stats['algorithm'] == algo]
        avg_colors = [df_algo[df_algo['graph'] == g]['avg_colors'].values[0] if len(df_algo[df_algo['graph'] == g]) > 0 else 0 for g in graphs]
        std_colors = [df_algo[df_algo['graph'] == g]['std_colors'].values[0] if len(df_algo[df_algo['graph'] == g]) > 0 else 0 for g in graphs]
        ax.bar(x + bar_index * bar_width, avg_colors, bar_width, yerr=std_colors, 
               label=algo, capsize=5, color=ALGO_COLORS.get(algo, '#888888'), alpha=0.85)
        bar_index += 1
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Color Count', fontweight='bold', fontsize=12)
    ax.set_title('Average Color Count Comparison (with Standard Deviation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 0.4)  # Center ticks
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    avg_colors_file = output_path / 'comparison_avg_colors.png'
    if avg_colors_file.exists():
        avg_colors_file.unlink()
    fig.savefig(avg_colors_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2/4 created: Average Color Count Comparison")
    
    # Plot 3: Average Execution Time (NO BEST KNOWN - doesn't apply)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_width = 0.25
    for i, algo in enumerate(algorithms):
        df_algo = df_stats[df_stats['algorithm'] == algo]
        avg_time = [df_algo[df_algo['graph'] == g]['avg_time'].values[0] if len(df_algo[df_algo['graph'] == g]) > 0 else 0 for g in graphs]
        ax.bar(x + i * algo_width, avg_time, algo_width, label=algo, 
               color=ALGO_COLORS.get(algo, '#888888'), alpha=0.85)
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Execution Time (s)', fontweight='bold', fontsize=12)
    ax.set_title('Execution Time Comparison (log scale)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + algo_width * (n_algos - 1) / 2)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, which='both')
    ax.set_yscale('log')
    plt.tight_layout()
    
    exec_time_file = output_path / 'comparison_execution_time.png'
    if exec_time_file.exists():
        exec_time_file.unlink()
    fig.savefig(exec_time_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3/4 created: Execution Time Comparison")
    
    # Plot 4: Average Conflict Count (NO BEST KNOWN - doesn't apply)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        df_algo = df_stats[df_stats['algorithm'] == algo]
        avg_conflicts = [df_algo[df_algo['graph'] == g]['avg_conflicts'].values[0] if len(df_algo[df_algo['graph'] == g]) > 0 else 0 for g in graphs]
        ax.bar(x + i * algo_width, avg_conflicts, algo_width, label=algo, 
               color=ALGO_COLORS.get(algo, '#888888'), alpha=0.85)
    
    ax.set_xlabel('Graph', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Conflict Count', fontweight='bold', fontsize=12)
    ax.set_title('Conflict Count Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + algo_width * (n_algos - 1) / 2)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    conflicts_file = output_path / 'comparison_conflicts.png'
    if conflicts_file.exists():
        conflicts_file.unlink()
    fig.savefig(conflicts_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4/4 created: Conflict Count Comparison")



def save_results_to_json(df_results, df_stats, output_dir, aco_params=None, best_known=None):
    """
    Save results and statistics to JSON file.
    
    Args:
        df_results: DataFrame with all test results
        df_stats: DataFrame with statistics
        output_dir: Directory to save JSON file
        aco_params: ACO parameters used in testing
        best_known: Best known results dictionary
    """
    output_path = Path(output_dir)
    
    # Convert DataFrames to dictionaries
    results_dict = df_results.to_dict(orient='records')
    stats_dict = df_stats.to_dict(orient='records')
    
    # Create complete data structure
    data = {
        'completed': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'aco_params': aco_params,
        'best_known': best_known if best_known else {},
        'results': results_dict,
        'statistics': stats_dict
    }
    
    json_file = output_path / 'comparison_results.json'
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Results cached to JSON (comparison_results.json)")


def load_results_from_json(output_dir):
    """
    Load results from JSON file if it exists.
    
    Args:
        output_dir: Directory containing JSON file
        
    Returns:
        Tuple of (df_results, df_stats, aco_params, best_known) or (None, None, None, None) if file doesn't exist
    """
    json_file = Path(output_dir) / 'comparison_results.json'
    
    if not json_file.exists():
        return None, None, None, None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data.get('completed', False):
            return None, None, None, None
        
        df_results = pd.DataFrame(data['results'])
        df_stats = pd.DataFrame(data['statistics'])
        aco_params = data.get('aco_params')
        best_known = data.get('best_known', {})
        
        print(f"✓ Loaded existing results from: {json_file}")
        print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
        print(f"  Total runs: {len(df_results)}")
        if best_known:
            print(f"  Best known solutions: {len(best_known)} graphs")
        
        return df_results, df_stats, aco_params, best_known
    
    except Exception as e:
        print(f"⚠ Error loading JSON file: {e}")
        return None, None, None, None


def run_comprehensive_testing(data_root, dataset_name, output_dir, num_repetitions=5, aco_params=None, force_rerun=False):
    """
    Main function to run comprehensive testing.
    
    Args:
        data_root: Root data directory path
        dataset_name: Dataset name ('tiny_dataset' or 'main_dataset')
        output_dir: Directory to save comparison results (inside study path)
        num_repetitions: Number of repetitions per algorithm (default: 5)
        aco_params: Optional ACO parameters
        force_rerun: If True, ignore cached results and rerun testing (default: False)
        
    Returns:
        Tuple of (results_df, stats_df)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataloader
    loader = GraphDataLoader(data_root, dataset_name)
    
    # Load best known results (required for testing)
    best_known = loader.load_best_known_results()
    
    # Load testing dataset
    test_graphs = loader.load_testing_dataset()
    
    # Try to load existing results
    if not force_rerun:
        print("Checking for existing test results...")
        df_results, df_stats, cached_aco_params, cached_best_known = load_results_from_json(output_dir)
        
        if df_results is not None and df_stats is not None:
            print("\n" + "=" * 80)
            print("USING CACHED RESULTS (set force_rerun=True to re-execute)")
            print("=" * 80)
            
            # Use cached best_known if available, otherwise use newly loaded
            bks_to_use = cached_best_known if cached_best_known else best_known
            
            # Regenerate per-algorithm plots from cached data
            print("\n" + "=" * 80)
            print("REGENERATING PER-ALGORITHM PLOTS FROM CACHED DATA")
            print("=" * 80)
            
            for algo in ['greedy', 'tabu', 'aco']:
                df_algo = df_results[df_results['algorithm'] == algo]
                algo_df = pd.DataFrame({
                    'graph': df_algo['graph'],
                    'colors': df_algo['color_count'],
                    'conflicts': df_algo['conflict_count'],
                    'time': df_algo['execution_time']
                })
                algo_best_params = cached_aco_params if algo == 'aco' and cached_aco_params else None
                generate_algorithm_plots(algo_df, algo, output_dir, algo_best_params)
            
            # Regenerate comparison plots from cached data
            print("\n" + "=" * 80)
            print("REGENERATING COMPARISON PLOTS FROM CACHED DATA")
            print("=" * 80)
            generate_comparison_plots(df_results, df_stats, output_dir, bks_to_use)
            
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(df_stats.to_string(index=False))
            
            return df_results, df_stats
    
    # Run new testing
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM TESTING")
    print("=" * 80)
    
    df_results = run_comprehensive_test(test_graphs, output_dir, num_repetitions, aco_params)
    
    print("\n" + "=" * 80)
    print("CALCULATING STATISTICS")
    print("=" * 80)
    
    df_stats = calculate_statistics(df_results, best_known)
    
    # Save to CSV
    stats_file = output_path / 'statistics_summary.csv'
    df_stats.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to CSV (statistics_summary.csv)")
    
    # Save to JSON
    save_results_to_json(df_results, df_stats, output_dir, aco_params, best_known)
    
    # Generate per-algorithm plots
    print("\n" + "=" * 80)
    print("GENERATING PER-ALGORITHM PLOTS")
    print("=" * 80)
    
    for algo in ['greedy', 'tabu', 'aco']:
        df_algo = df_results[df_results['algorithm'] == algo]
        algo_df = pd.DataFrame({
            'graph': df_algo['graph'],
            'colors': df_algo['color_count'],
            'conflicts': df_algo['conflict_count'],
            'time': df_algo['execution_time']
        })
        algo_best_params = aco_params if algo == 'aco' else None
        generate_algorithm_plots(algo_df, algo, output_dir, algo_best_params)
    
    # Generate comparison plots across all algorithms
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    generate_comparison_plots(df_results, df_stats, output_dir, best_known)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(df_stats.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    
    return df_results, df_stats


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python comprehensive_testing.py <data_root> <dataset_name> <output_dir> [num_repetitions]")
        print("Example: python comprehensive_testing.py /path/to/data main_dataset /path/to/output 5")
        sys.exit(1)
    
    data_root = sys.argv[1]
    dataset_name = sys.argv[2]
    output_dir = sys.argv[3]
    num_reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    run_comprehensive_testing(data_root, dataset_name, output_dir, num_reps)
