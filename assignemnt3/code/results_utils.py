"""
Results visualization and export utilities for ACO hyperparameter tuning.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from visualization_utils import save_colored_graph_image


def visualize_testing_results(testing_results, study_name, data_root):
    """
    Create and save visualization of testing results in study/testing folder.
    Also saves colored graph visualizations as PNG files.
    
    Args:
        testing_results: Dictionary containing test results for each graph
        study_name: Name of the study for file naming
        data_root: Path to data root directory
    
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

    # Save summary figure in testing folder
    testing_results_fig_path = testing_path / 'summary.png'
    plt.savefig(testing_results_fig_path, dpi=300, bbox_inches='tight')
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
