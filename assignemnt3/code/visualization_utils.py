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


def generate_trial_visualizations(trial_number: int, graph_results: Dict, trial_dir: Path, params: Dict = None):
    """
    Generate visualization plots for a trial's results.
    
    Args:
        trial_number: Trial number
        graph_results: Dictionary mapping graph name to result dict
        trial_dir: Directory to save figures
        params: Trial parameters (iterations, ant_count, etc.)
    """
    if not graph_results:
        return
    
    graph_names = list(graph_results.keys())
    colors = [graph_results[g]['color_count'] for g in graph_names]
    times = [graph_results[g]['elapsed_time'] for g in graph_names]
    conflicts = [graph_results[g].get('conflict_count', 0) for g in graph_names]  # ACO doesn't return conflicts (always 0)
    
    # Metric 1: Color counts per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(graph_names)), colors, color='steelblue')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Number of Colors', fontsize=12)
    ax.set_title(f'Trial {trial_number}: Color Count per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add color count values inside bars
    max_colors = max(colors) if colors else 1
    threshold = max_colors * 0.25
    
    for i, v in enumerate(colors):
        if v < threshold:
            ax.text(i, v + max_colors*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(trial_dir / "color_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metric 2: Execution time per graph (separate image)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(range(len(graph_names)), times, color='coral')
    ax.set_xlabel('Graph', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Trial {trial_number}: Execution Time per Graph', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add labels with time and iterations only
    max_time = max(times) if times else 1.0
    threshold = max_time * 0.25  # 25% of max height
    
    for i, v in enumerate(times):
        label_parts = [f'{v:.2f}s']
        if params and 'iterations' in params:
            label_parts.append(f"iter={params['iterations']}")
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
    
    # Add conflict count values inside bars
    max_conflicts = max(conflicts) if conflicts else 1
    threshold = max_conflicts * 0.25
    
    for i, v in enumerate(conflicts):
        if v < threshold:
            ax.text(i, v + max_conflicts*0.02, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
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
                conflict_count=graph_data.get('conflict_count', 0),  # Default to 0 since constructive approach has no conflicts
                save_path=graph_path,
                node_size=50
            )
            saved_graphs.append(f"graph_{graph_name}.png")
    
    print(f"    Saved trial visualizations: 3 metric plots, {len(saved_graphs)} graph plots")


def save_colored_graph_image(graph, solution, graph_name, color_count, conflict_count, save_path, node_size=50):
    """
    Save a colored graph visualization to a file using NetworkX.
    Completely isolated implementation to prevent any figure leakage or display issues.
    
    Args:
        graph: NetworkX graph object
        solution: Dictionary mapping node to color
        graph_name: Name of the graph for title
        color_count: Number of colors used
        conflict_count: Number of conflicts
        save_path: Path object where to save the image
        node_size: Size of nodes in visualization
    """
    # Ensure Agg backend BEFORE any imports
    import matplotlib
    matplotlib.use('Agg', force=True)
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx
    
    # Ensure completely clean state
    plt.close('all')
    plt.ioff()
    
    # Create NEW figure with unique number to avoid any conflicts
    import random
    fig_num = random.randint(10000, 99999)
    fig = plt.figure(num=fig_num, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(111)
    
    try:
        # Create color map
        unique_colors = len(set(solution.values()))
        cmap = cm.get_cmap('tab20', unique_colors)
        
        # Map node colors
        node_colors = [cmap(solution.get(node, 0)) for node in graph.nodes()]
        
        # Use spring layout for visualization with fixed seed for consistency
        pos = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
        
        # Draw graph components
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=node_size, ax=ax, alpha=0.9)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Set title and styling
        ax.set_title(f"{graph_name}\\nColors: {color_count}, Conflicts: {conflict_count}",
                    fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
        ax.margins(0.1)
        
        # Save with tight layout
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight', 
                   format='png', facecolor='white', edgecolor='none')
        
    except Exception as e:
        print(f"Warning: Error saving graph {graph_name}: {e}")
        raise
    finally:
        # Aggressive cleanup
        plt.close(fig)
        plt.close(fig_num)
        plt.close('all')
        # Clear any remaining figures
        for i in plt.get_fignums():
            plt.close(i)


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


def plot_slice(study: optuna.Study, figures_path: Path):
    """Plot slice plot showing parameter effects with dynamic layout."""
    filepath = figures_path / "slice.png"
    try:
        # Get parameter names
        params = list(study.best_params.keys())
        if not params:
            print("No parameters to plot")
            return
        
        # Calculate subplot layout dynamically based on number of parameters
        n_params = len(params)
        if n_params <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
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


def plot_timeline(study: optuna.Study, figures_path: Path):
    """
    Plot timeline of trial execution with key parameter annotations.
    Shows total time, ant count, and iterations for each trial.
    """
    filepath = figures_path / "timeline.png"
    try:
        # Generate base timeline plot from Optuna - returns matplotlib Axes object
        ax = optuna.visualization.matplotlib.plot_timeline(study)
        fig = ax.figure  # Get the figure from the axes
        
        # Add parameter annotations to each trial
        trials = study.trials
        for i, trial in enumerate(trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Collect only the 3 key parameters: total time, ant_count, iterations
                params = trial.params
                param_text_parts = []
                
                # Calculate total time
                if trial.datetime_complete and trial.datetime_start:
                    total_time = (trial.datetime_complete - trial.datetime_start).total_seconds()
                    param_text_parts.append(f"{total_time:.1f}s")
                
                if 'ant_count' in params:
                    param_text_parts.append(f"ants={params['ant_count']}")
                
                if 'iterations' in params:
                    param_text_parts.append(f"iter={params['iterations']}")
                
                # Create annotation text and place inside the bar
                if param_text_parts:
                    param_text = '\n'.join(param_text_parts)
                    # Calculate midpoint of the trial bar
                    bar_start = trial.datetime_start
                    bar_end = trial.datetime_complete
                    bar_mid = bar_start + (bar_end - bar_start) / 2
                    
                    ax.text(bar_mid, i, param_text, 
                            va='center', ha='center', fontsize=7, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='black', alpha=0.6))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Timeline plot saved to: {filepath.name}")
    except Exception as e:
        print(f"Could not plot timeline: {e}")
        import traceback
        traceback.print_exc()

