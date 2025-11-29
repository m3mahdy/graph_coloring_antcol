"""
Results visualization and export utilities for ACO hyperparameter tuning.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_testing_results(testing_results, study_name, data_root):
    """
    Create and save visualization of testing results.
    
    Args:
        testing_results: Dictionary containing test results for each graph
        study_name: Name of the study for file naming
        data_root: Path to data root directory
    
    Returns:
        Path: Path to the saved figure
    """
    # Extract data for visualization
    graph_names = list(testing_results.keys())
    color_counts = [testing_results[g]['color_count'] for g in graph_names]
    conflict_counts = [testing_results[g]['conflict_count'] for g in graph_names]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Color counts per graph
    axes[0].bar(range(len(graph_names)), color_counts, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Graph', fontsize=12)
    axes[0].set_ylabel('Number of Colors Used', fontsize=12)
    axes[0].set_title('Color Count per Testing Graph', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(graph_names)))
    axes[0].set_xticklabels(graph_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(color_counts):
        axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

    # Plot 2: Conflict counts per graph
    axes[1].bar(range(len(graph_names)), conflict_counts, color='coral', alpha=0.7)
    axes[1].set_xlabel('Graph', fontsize=12)
    axes[1].set_ylabel('Number of Conflicts', fontsize=12)
    axes[1].set_title('Conflict Count per Testing Graph', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(graph_names)))
    axes[1].set_xticklabels(graph_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(conflict_counts):
        axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save figure
    figures_path = Path(data_root) / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    testing_results_fig_path = figures_path / f'{study_name}_testing_results.png'
    plt.savefig(testing_results_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return testing_results_fig_path


def export_results(testing_results, best_params, study_name, data_root):
    """
    Export testing results and best parameters to JSON and CSV files.
    
    Args:
        testing_results: Dictionary containing test results for each graph
        best_params: Dictionary of best hyperparameters found
        study_name: Name of the study for file naming
        data_root: Path to data root directory
    
    Returns:
        dict: Dictionary containing paths to all exported files
    """
    data_root = Path(data_root)
    results_dir = data_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save testing results to JSON
    testing_results_path = results_dir / f'{study_name}_testing_results.json'
    with open(testing_results_path, 'w') as f:
        json.dump(testing_results, f, indent=2)
    
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
    summary_csv_path = results_dir / f'{study_name}_testing_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame([best_params])
    best_params_csv_path = results_dir / f'{study_name}_best_params.csv'
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


def print_file_locations(study_name, data_root, exported_files, figure_path):
    """
    Print a summary of all file locations.
    
    Args:
        study_name: Name of the study
        data_root: Path to data root directory
        exported_files: Dictionary containing paths to exported files
        figure_path: Path to the testing results figure
    """
    data_root = Path(data_root)
    
    print("\n" + "=" * 70)
    print("FILE LOCATIONS")
    print("=" * 70)
    print("\nStudy Files:")
    print(f"  Study Log: {data_root / 'studies' / f'{study_name}.log'}")
    print(f"  Best Params JSON: {data_root / 'studies' / f'{study_name}_best_params.json'}")

    print("\nResults:")
    print(f"  Testing Results JSON: {exported_files['testing_results_json']}")
    print(f"  Testing Summary CSV: {exported_files['testing_summary_csv']}")
    print(f"  Best Params CSV: {exported_files['best_params_csv']}")

    print("\nFigures:")
    print(f"  Optimization History: {data_root / 'figures' / f'{study_name}_optimization_history.png'}")
    print(f"  Parameter Importances: {data_root / 'figures' / f'{study_name}_param_importances.png'}")
    print(f"  Parallel Coordinates: {data_root / 'figures' / f'{study_name}_parallel_coordinate.png'}")
    print(f"  Testing Results: {figure_path}")
    print("=" * 70)
