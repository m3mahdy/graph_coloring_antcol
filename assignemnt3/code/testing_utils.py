"""
Testing Results Export and Visualization for ACO Hyperparameter Tuning.

This module handles TESTING phase operations ONLY:
- Exports testing results to JSON/CSV files
- Creates summary visualizations for testing results
- Saves colored graph visualizations for testing graphs
- Prints summary statistics and file locations

Note: This is separate from visualization_utils.py which handles:
- Trial visualizations during optimization
- Study-level visualizations (history, importances, etc.)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from visualization_utils import save_colored_graph_image


def generate_algorithm_plots(results_df, algorithm_name, output_dir, best_params=None):
    """
    Generate 3 metric plots for a single algorithm's testing results.
    Saves plots in algorithm-specific folder: output_dir/algorithm_name/
    
    Args:
        results_df: DataFrame with columns [graph, colors, conflicts, time]
        algorithm_name: Name of algorithm (greedy, tabu, aco)
        output_dir: Base output directory path
        best_params: Optional best parameters (for ACO iteration display)
        
    Returns:
        Path: Path to algorithm's results folder
    """
    # Create algorithm folder
    algo_path = Path(output_dir) / algorithm_name
    algo_path.mkdir(parents=True, exist_ok=True)
    
    graph_names = results_df['graph'].tolist()
    color_counts = results_df['colors'].tolist()
    conflict_counts = results_df['conflicts'].tolist()
    times = results_df['time'].tolist()
    
    # Plot 1: Color counts per graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), color_counts, color='steelblue')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Number of Colors', fontsize=12)
    ax.set_title(f'{algorithm_name.upper()}: Color Count per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    max_colors = max(color_counts) if color_counts else 1
    threshold = max_colors * 0.25
    
    for i, v in enumerate(color_counts):
        if v < threshold:
            ax.text(i, v + max_colors*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    color_count_file = algo_path / "color_count.png"
    if color_count_file.exists():
        color_count_file.unlink()
    plt.savefig(color_count_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Execution time per graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(len(graph_names)), times, color='coral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'{algorithm_name.upper()}: Execution Time per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    max_time = max(times) if times else 1.0
    threshold = max_time * 0.25
    
    for i, v in enumerate(times):
        label_parts = [f'{v:.2f}s']
        if best_params and 'iterations' in best_params and algorithm_name == 'aco':
            test_iterations = int(best_params['iterations'] * 1.5)
            label_parts.append(f"iter={test_iterations}")
        label = '\n'.join(label_parts)
        
        if v < threshold:
            ax.text(i, v + max_time*0.02, label, ha='center', va='bottom', fontsize=8, 
                    fontweight='bold', color='black')
        else:
            ax.text(i, v/2, label, ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    exec_time_file = algo_path / "execution_time.png"
    if exec_time_file.exists():
        exec_time_file.unlink()
    plt.savefig(exec_time_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Conflict counts per graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), conflict_counts, color='lightcoral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Conflicts', fontsize=12)
    ax.set_title(f'{algorithm_name.upper()}: Conflicts per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    max_conflicts = max(conflict_counts) if conflict_counts else 1
    threshold = max_conflicts * 0.25
    
    for i, v in enumerate(conflict_counts):
        if v < threshold:
            ax.text(i, v + max_conflicts*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    conflicts_file = algo_path / "conflicts.png"
    if conflicts_file.exists():
        conflicts_file.unlink()
    plt.savefig(conflicts_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated {algorithm_name} plots: 3 metrics (color count, execution time, conflicts)")
    
    return algo_path


def visualize_testing_results(testing_results, study_name, data_root, best_params=None):
    """
    Create and save visualization of testing results in study/testing folder.
    Generates the same visualization format as trial results:
    - color_count.png: Bar chart of colors per graph
    - execution_time.png: Bar chart of execution time per graph
    - conflicts.png: Bar chart of conflicts per graph
    - graph_*.png: Colored graph visualizations
    
    Args:
        testing_results: Dictionary containing test results for each graph
        study_name: Name of the study for file naming
        data_root: Path to data root directory
        best_params: Best parameters from optimization (for execution time plot)
    
    Returns:
        Path: Path to the testing folder
    """
    # Create testing folder inside study folder
    testing_path = Path(data_root) / 'studies' / study_name / 'testing'
    testing_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data for visualization
    graph_names = list(testing_results.keys())
    color_counts = [testing_results[g]['color_count'] for g in graph_names]
    conflict_counts = [testing_results[g]['conflict_count'] for g in graph_names]
    times = [testing_results[g]['elapsed_time'] for g in graph_names]

    # Metric 1: Color counts per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), color_counts, color='steelblue')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Number of Colors', fontsize=12)
    ax.set_title('Testing: Color Count per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add color count values inside bars
    max_colors = max(color_counts) if color_counts else 1
    threshold = max_colors * 0.25
    
    for i, v in enumerate(color_counts):
        if v < threshold:
            ax.text(i, v + max_colors*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(testing_path / "color_count.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Metric 2: Execution time per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(range(len(graph_names)), times, color='coral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Testing: Execution Time per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add labels with time and iterations only
    max_time = max(times) if times else 1.0
    threshold = max_time * 0.25
    
    for i, v in enumerate(times):
        label_parts = [f'{v:.2f}s']
        if best_params and 'iterations' in best_params:
            # Show 1.5x iterations used in testing
            test_iterations = int(best_params['iterations'] * 1.5)
            label_parts.append(f"iter={test_iterations}")
        label = '\n'.join(label_parts)
        
        # Place label above bar if it's less than 25% of max, else inside
        if v < threshold:
            ax.text(i, v + max_time*0.02, label, ha='center', va='bottom', fontsize=8, 
                    fontweight='bold', color='black')
        else:
            ax.text(i, v/2, label, ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(testing_path / "execution_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Metric 3: Conflict counts per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), conflict_counts, color='lightcoral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Conflicts', fontsize=12)
    ax.set_title('Testing: Conflicts per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add conflict count values inside bars
    max_conflicts = max(conflict_counts) if conflict_counts else 1
    threshold = max_conflicts * 0.25
    
    for i, v in enumerate(conflict_counts):
        if v < threshold:
            ax.text(i, v + max_conflicts*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(testing_path / "conflicts.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save colored graph visualizations
    for graph_name in graph_names:
        graph_data = testing_results[graph_name]
        if 'graph' in graph_data and 'best_solution' in graph_data:
            graph_path = testing_path / f"graph_{graph_name}.png"
            save_colored_graph_image(
                graph=graph_data['graph'],
                solution=graph_data['best_solution'],
                graph_name=graph_name,
                color_count=graph_data['color_count'],
                conflict_count=graph_data['conflict_count'],
                save_path=graph_path,
                node_size=100
            )
    
    print(f"    Saved testing visualizations: 3 metric plots, {len(graph_names)} graph plots")
    
    return testing_path


def export_results(testing_results, best_params, study_name, data_root):
    """
    Export testing results and best parameters to JSON and CSV files in testing folder.
    
    Args:
        testing_results: Dictionary containing test results for each graph
        best_params: Dictionary of best hyperparameters found
        study_name: Name of the study for file naming
        data_root: Path to data root directory
    
    Returns:
        dict: Dictionary containing paths to all exported files
    """
    data_root = Path(data_root)
    testing_dir = data_root / 'studies' / study_name / 'testing'
    testing_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean testing results (remove non-JSON serializable data)
    clean_results = {}
    for graph_name, result in testing_results.items():
        clean_results[graph_name] = {
            k: v for k, v in result.items() 
            if k not in ['graph', 'best_solution']
        }
    
    # Save testing results to JSON
    testing_results_path = testing_dir / 'results.json'
    with open(testing_results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for graph_name, result in testing_results.items():
        summary_data.append({
            'Graph': graph_name,
            'Colors Used': result['color_count'],
            'Conflicts': result['conflict_count'],
            'Iterations': result['iterations_used']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_csv_path = testing_dir / 'summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame([best_params])
    best_params_csv_path = testing_dir / 'best_params.csv'
    best_params_df.to_csv(best_params_csv_path, index=False)
    
    return {
        'testing_results_json': testing_results_path,
        'testing_summary_csv': summary_csv_path,
        'best_params_csv': best_params_csv_path,
        'summary_df': summary_df
    }


def print_summary_statistics(summary_df):
    """
    Print formatted summary statistics for testing results.
    
    Args:
        summary_df: DataFrame containing testing results summary
    """
    print("\n" + "=" * 70)
    print("TESTING RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Average Colors Used: {summary_df['Colors Used'].mean():.2f}")
    print(f"Average Conflicts: {summary_df['Conflicts'].mean():.2f}")
    print(f"Total Graphs Tested: {len(summary_df)}")
    print(f"Graphs with Zero Conflicts: {(summary_df['Conflicts'] == 0).sum()}")
    print("=" * 70)


def print_file_locations(study_name, data_root, exported_files, testing_path):
    """
    Print a summary of all file locations.
    
    Args:
        study_name: Name of the study
        data_root: Path to data root directory
        exported_files: Dictionary containing paths to exported files
        testing_path: Path to the testing folder
    """
    data_root = Path(data_root)
    study_folder = data_root / 'studies' / study_name
    
    print("\n" + "=" * 70)
    print("FILE LOCATIONS")
    print("=" * 70)
    print(f"\nStudy Folder: {study_folder}")
    print("\nStudy Files:")
    print(f"  Study Log: {study_folder / f'{study_name}.log'}")
    print(f"  Study Summary: {study_folder / f'{study_name}_summary.json'}")

    print("\nTesting Results (testing/):")
    print(f"  Results JSON: {exported_files['testing_results_json']}")
    print(f"  Summary CSV: {exported_files['testing_summary_csv']}")
    print(f"  Best Params CSV: {exported_files['best_params_csv']}")
    print(f"  Summary Figure: {testing_path / 'summary.png'}")
    print(f"  Colored Graphs: {testing_path / 'graph_*.png'}")

    print("\nStudy Figures (figures/):")
    print(f"  Optimization History: {study_folder / 'figures' / 'history.png'}")
    print(f"  Parameter Importances: {study_folder / 'figures' / 'importances.png'}")
    print(f"  Parallel Coordinates: {study_folder / 'figures' / 'parallel.png'}")
    print(f"  Contour Plot: {study_folder / 'figures' / 'contour.png'}")
    print(f"  Slice Plot: {study_folder / 'figures' / 'slice.png'}")
    
    print(f"\nTrial Figures (results/trial_XXXX/):")
    print(f"  Trial Data: {study_folder / 'results' / 'trial_0000' / 'trial_results.json'}")
    print(f"  Color Count: {study_folder / 'results' / 'trial_0000' / 'color_count.png'}")
    print(f"  Execution Time: {study_folder / 'results' / 'trial_0000' / 'execution_time.png'}")
    print(f"  Conflicts: {study_folder / 'results' / 'trial_0000' / 'conflicts.png'}")
    print(f"  Colored Graphs: {study_folder / 'results' / 'trial_0000' / 'graph_*.png'}")
    print("=" * 70)
