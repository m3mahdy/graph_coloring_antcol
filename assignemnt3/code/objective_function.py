"""
Objective function for ACO hyperparameter tuning with Optuna.
"""
import time
import json
from pathlib import Path
from datetime import datetime

# Global storage for graph visualization data (not JSON serializable)
_trial_graph_viz_data = {}


def aco_objective_function(trial, params, tuning_graphs, aco_class, verbose, recovery_dir=None, n_startup_trials=10, tabu_best_values=None):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        params: Dictionary of hyperparameters suggested by Optuna
        tuning_graphs: List of tuples (graph_name, graph) from tuning dataset
        aco_class: ACOGraphColoring class (not instantiated)
        verbose: Whether to show detailed ACO progress (overridden by params if present)
        recovery_dir: Optional directory path for saving intermediate results
        n_startup_trials: Number of random exploration trials before optimization
        tabu_best_values: Dictionary mapping graph names to tabu search best color counts
    
    Returns:
        float: Sum of color counts across all tuning graphs (to be minimized)
    """
    graph_results = {}
    total_color_count = 0
    
    # Use provided tabu best values or default to empty dict
    if tabu_best_values is None:
        tabu_best_values = {}
    
    # Store graph data separately for visualization (not in trial attrs - not JSON serializable)
    graph_data_for_viz = {}
    
    # Setup recovery file path if recovery_dir provided
    recovery_file = None
    completed_graphs = set()
    if recovery_dir:
        recovery_path = Path(recovery_dir)
        recovery_path.mkdir(parents=True, exist_ok=True)
        recovery_file = recovery_path / f"trial_{trial.number}_recovery.json"
        
        # Check if there's existing partial progress for this trial
        if recovery_file.exists():
            try:
                with open(recovery_file, 'r') as f:
                    recovery_data = json.load(f)
                    
                # Verify parameters match (ensure we're resuming the same trial)
                if recovery_data.get('params') == params:
                    graph_results = recovery_data.get('graph_results', {})
                    total_color_count = recovery_data.get('total_color_count', 0)
                    completed_graphs = set(graph_results.keys())
                    
                    print(f"\n{'='*70}")
                    print(f"🔄 Resuming Trial {trial.number} from previous interruption")
                    print(f"✓ Already completed {len(completed_graphs)}/{len(tuning_graphs)} graphs")
                    print(f"✓ Current total color count: {total_color_count}")
                    print(f"{'='*70}")
                else:
                    print(f"\n⚠ Recovery file found but parameters don't match - starting fresh")
                    completed_graphs = set()
            except Exception as e:
                print(f"\n⚠ Error loading recovery file: {e} - starting fresh")
                completed_graphs = set()
    
    # Print trial header
    if not completed_graphs:  # Only print if starting fresh
        print(f"\n{'='*70}")
        if trial.number < n_startup_trials:
            print(f"Trial {trial.number}: Random Exploration (Startup {trial.number + 1}/{n_startup_trials})")
        else:
            print(f"Trial {trial.number}: TPE Optimization")
        print(f"{'='*70}")
        print("Parameters:")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value}")
        print(f"{'-'*70}")
    
    # Iterate through all graphs in the tuning dataset
    for idx, (graph_name, graph) in enumerate(tuning_graphs, 1):
        # Skip already completed graphs
        if graph_name in completed_graphs:
            print(f"\nGraph {idx}/{len(tuning_graphs)}: {graph_name} - ⏭ Skipped (already completed)")
            continue
        # Get tabu best value for this graph
        tabu_best = tabu_best_values.get(graph_name)
        tabu_info = f", tabu_best={tabu_best}" if tabu_best else ""
        print(f"\nGraph {idx}/{len(tuning_graphs)}: {graph_name} (nodes={len(graph.nodes())}, edges={len(graph.edges())}{tabu_info})")
        
        # Create ACO instance with all suggested hyperparameters
        aco = aco_class(graph=graph, **params)
        
        # Track execution time for this graph
        start_time = time.time()
        result = aco.run()
        elapsed_time = time.time() - start_time
        
        # Store JSON-serializable results for trial attributes
        graph_results[graph_name] = {
            'graph_size': len(graph.nodes()),
            'color_count': result['color_count'],
            'iterations_used': result['iterations'],
            'elapsed_time': elapsed_time,
            'solution': {str(k): int(v) for k, v in result['solution'].items()} if result.get('solution') else {}
        }
        
        # Store graph separately for visualization (not JSON serializable)
        graph_data_for_viz[graph_name] = {
            'graph': graph,
            'color_count': result['color_count'],
            'solution': result.get('solution')
        }
        
        # Accumulate total color count for objective
        total_color_count += result['color_count']
        
        # Enhanced output
        tabu_best = tabu_best_values.get(graph_name)
        print(f"  ✓ Result: {result['color_count']} colors in {result['iterations']} iterations")
        if tabu_best:
            diff = result['color_count'] - tabu_best
            if diff == 0:
                print(f"  🎯 Perfect Match! ACO: {result['color_count']} = Tabu Best: {tabu_best}")
            elif diff > 0:
                print(f"  ⚠️ ACO: {result['color_count']} vs Tabu Best: {tabu_best} (+{diff} colors)")
            else:
                print(f"  🏆 Better! ACO: {result['color_count']} vs Tabu Best: {tabu_best} ({diff} colors)")
        print(f"  ⏱  Time: {elapsed_time:.2f}s")
        
        # Save intermediate results after each graph (for recovery)
        if recovery_file:
            recovery_data = {
                'trial_number': trial.number,
                'params': params,
                'graph_results': graph_results,
                'total_color_count': total_color_count,
                'completed_graphs': list(graph_results.keys()),
                'timestamp': datetime.now().isoformat()
            }
            try:
                with open(recovery_file, 'w') as f:
                    json.dump(recovery_data, f, indent=2)
            except Exception as e:
                print(f"  ⚠ Warning: Could not save recovery data: {e}")
    
    print(f"\n{'-'*70}")
    print(f"Trial {trial.number} Summary:")
    print(f"  Total colors across all graphs: {total_color_count}")
    print(f"  Average colors per graph: {total_color_count / len(tuning_graphs):.2f}")
    print(f"{'='*70}\n")
    
    # Store only JSON-serializable data in trial user attributes
    trial.set_user_attr('graph_results', graph_results)
    
    # Store graph data in global dict for callback access
    _trial_graph_viz_data[trial.number] = graph_data_for_viz
    
    # Clean up recovery file on successful completion
    if recovery_file and recovery_file.exists():
        try:
            recovery_file.unlink()
            print(f"✓ Recovery file cleaned up")
        except Exception as e:
            print(f"⚠ Warning: Could not delete recovery file: {e}")
    
    return total_color_count
