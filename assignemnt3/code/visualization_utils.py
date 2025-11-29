"""
Visualization utilities for ACO hyperparameter tuning.
Handles all plotting operations for trials and study results.
"""

import optuna
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy
from pathlib import Path
from typing import Dict


def generate_trial_visualizations(trial_number: int, graph_results: Dict, trial_dir: Path):
    """
    Generate visualization plots for a trial's results.
    
    Args:
        trial_number: Trial number
        graph_results: Dictionary mapping graph name to result dict
        trial_dir: Directory to save figures
    """
    if not graph_results:
        return
    
    graph_names = list(graph_results.keys())
    colors = [graph_results[g]['color_count'] for g in graph_names]
    times = [graph_results[g]['elapsed_time'] for g in graph_names]
    conflicts = [graph_results[g]['conflict_count'] for g in graph_names]
    
    # Metric 1: Color counts per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), colors, color='steelblue')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Number of Colors', fontsize=12)
    ax.set_title(f'Trial {trial_number}: Color Count per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(colors):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(trial_dir / "color_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metric 2: Execution time per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), times, color='coral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Trial {trial_number}: Execution Time per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        ax.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(trial_dir / "execution_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metric 3: Conflict counts per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), conflicts, color='lightcoral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Conflicts', fontsize=12)
    ax.set_title(f'Trial {trial_number}: Conflicts per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(conflicts):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(trial_dir / "conflicts.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Graph coloring visualizations (each graph in separate file)
    saved_graphs = []
    for graph_name in graph_names:
        graph_data = graph_results[graph_name]
        if 'graph' in graph_data and 'solution' in graph_data:
            graph_path = trial_dir / f"graph_{graph_name}.png"
            save_colored_graph_image(
                graph=graph_data['graph'],
                solution=graph_data['solution'],
                graph_name=graph_name,
                color_count=graph_data['color_count'],
                conflict_count=graph_data['conflict_count'],
                save_path=graph_path,
                node_size=50
            )
            saved_graphs.append(f"graph_{graph_name}.png")
    
    print(f"    Saved trial visualizations: 3 metric plots, {len(saved_graphs)} graph plots")


def save_colored_graph_image(graph, solution, graph_name, color_count, conflict_count, save_path, node_size=50):
    """
    Save a colored graph visualization to a file using NetworkX.
    Creates a fresh figure, plots the graph, saves it, and cleans up.
    
    Args:
        graph: NetworkX graph object
        solution: Dictionary mapping node to color
        graph_name: Name of the graph for title
        color_count: Number of colors used
        conflict_count: Number of conflicts
        save_path: Path object where to save the image
        node_size: Size of nodes in visualization
    """
    # Create fresh figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Create color map
    unique_colors = len(set(solution.values()))
    cmap = cm.get_cmap('tab20', unique_colors)
    
    # Map node colors
    node_colors = [cmap(solution.get(node, 0)) for node in graph.nodes()]
    
    # Use spring layout for visualization
    pos = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
    
    # Draw graph using NetworkX
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_size, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, ax=ax)
    
    # Set title and remove axis
    ax.set_title(f"{graph_name}\nColors: {color_count}, Conflicts: {conflict_count}",
                fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Save and cleanup
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_colored_graph(ax, graph, solution, graph_name, color_count, conflict_count):
    """
    Plot a colored graph on given axis using NetworkX layout.
    
    Args:
        ax: Matplotlib axis
        graph: NetworkX graph
        solution: Dictionary mapping node to color
        graph_name: Name of the graph
        color_count: Number of colors used
        conflict_count: Number of conflicts
    """
    # Create color map
    unique_colors = len(set(solution.values()))
    cmap = cm.get_cmap('tab20', unique_colors)
    
    # Map node colors
    node_colors = [cmap(solution.get(node, 0)) for node in graph.nodes()]
    
    # Use spring layout for visualization
    pos = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
    
    # Draw graph using NetworkX
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=50, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, ax=ax)
    
    ax.set_title(f"{graph_name}\nColors: {color_count}, Conflicts: {conflict_count}",
                fontsize=11, fontweight='bold')
    ax.axis('off')


def plot_study_history(study: optuna.Study, figures_path: Path):
    """Plot optimization history."""
    filepath = figures_path / "history.png"
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization history saved to: {filepath.name}")


def plot_param_importances(study: optuna.Study, figures_path: Path):
    """Plot parameter importance."""
    if len(study.trials) < 2:
        print("Not enough trials to compute parameter importances")
        return
    
    filepath = figures_path / "importances.png"
    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter importances saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")


def plot_parallel_coordinate(study: optuna.Study, figures_path: Path):
    """Plot parallel coordinate plot."""
    filepath = figures_path / "parallel.png"
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parallel coordinate plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")


def plot_contour(study: optuna.Study, figures_path: Path):
    """Plot contour plot for parameter relationships."""
    if len(study.trials) < 3:
        print("Not enough trials for contour plot (need at least 3)")
        return
    
    filepath = figures_path / "contour.png"
    try:
        fig = optuna.visualization.matplotlib.plot_contour(study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Contour plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot contour: {e}")


def plot_slice(study: optuna.Study, figures_path: Path):
    """Plot slice plot showing parameter effects in 2 rows x 3 columns."""
    filepath = figures_path / "slice.png"
    try:
        # Get parameter names
        params = list(study.best_params.keys())
        if not params:
            print("No parameters to plot")
            return
        
        # Calculate subplot layout (2 rows, 3 columns)
        n_params = len(params)
        n_cols = 3
        n_rows = 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
        
        # Plot each parameter
        for idx, param in enumerate(params):
            if idx < len(axes):
                ax = axes[idx]
                # Get trial data
                trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if trials:
                    values = [t.params.get(param) for t in trials]
                    objectives = [t.value for t in trials]
                    ax.scatter(values, objectives, alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
                    ax.set_xlabel(param, fontsize=11, fontweight='bold')
                    ax.set_ylabel('Objective Value', fontsize=11)
                    ax.set_title(f'Slice plot of {param}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Slice plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot slice: {e}")


def plot_edf(study: optuna.Study, figures_path: Path):
    """
    Plot empirical distribution function (EDF).
    
    EDF shows the cumulative distribution of objective values across trials.
    It helps visualize the probability of achieving a certain objective value or better.
    X-axis: Objective value | Y-axis: Proportion of trials with value ≤ x
    """
    filepath = figures_path / "edf.png"
    try:
        fig = optuna.visualization.matplotlib.plot_edf(study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"EDF plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot EDF: {e}")


def plot_timeline(study: optuna.Study, figures_path: Path):
    """Plot timeline of trial execution."""
    filepath = figures_path / "timeline.png"
    try:
        fig = optuna.visualization.matplotlib.plot_timeline(study)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Timeline plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot timeline: {e}")
