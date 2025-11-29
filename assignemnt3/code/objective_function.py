"""
Objective function for ACO hyperparameter tuning with Optuna.
"""
import time

# Global storage for graph visualization data (not JSON serializable)
_trial_graph_viz_data = {}


def aco_objective_function(trial, params, tuning_graphs, aco_class, verbose):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        params: Dictionary of hyperparameters suggested by Optuna
        tuning_graphs: List of tuples (graph_name, graph) from tuning dataset
        aco_class: ACOGraphColoring class (not instantiated)
        verbose: Whether to show detailed ACO progress (overridden by params if present)
    
    Returns:
        float: Sum of color counts across all tuning graphs (to be minimized)
    """
    graph_results = {}
    total_color_count = 0
    
    # Store graph data separately for visualization (not in trial attrs - not JSON serializable)
    graph_data_for_viz = {}
    
    # Print trial header
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*70}")
    print(f"{'-'*70}")
    
    # Iterate through all graphs in the tuning dataset
    for idx, (graph_name, graph) in enumerate(tuning_graphs, 1):
        print(f"\nGraph {idx}/{len(tuning_graphs)}: {graph_name} (nodes={len(graph.nodes())}, edges={len(graph.edges())})")
        
        # Create ACO instance with all suggested hyperparameters
        aco = aco_class(graph=graph, **params)
        
        # Track execution time for this graph
        start_time = time.time()
        result = aco.run()
        elapsed_time = time.time() - start_time
        
        # Store JSON-serializable results for trial attributes
        graph_results[graph_name] = {
            'initial_num_colors': aco.initial_num_colors,
            'graph_size': len(graph.nodes()),
            'color_count': result['color_count'],
            'iterations_used': result['iterations'],
            'elapsed_time': elapsed_time
        }
        
        # Store graph separately for visualization (not JSON serializable)
        # Note: ACO does not return solution dict, only color count
        graph_data_for_viz[graph_name] = {
            'graph': graph,
            'color_count': result['color_count']
        }
        
        # Accumulate total color count for objective
        total_color_count += result['color_count']
        
        # Enhanced output
        print(f"  ✓ Result: {result['color_count']} colors used in {result['iterations']} iterations")
        print(f"  ✓ Time: {elapsed_time:.2f}s")
    
    print(f"\n{'-'*70}")
    print(f"Trial {trial.number} Summary:")
    print(f"  Total colors across all graphs: {total_color_count}")
    print(f"  Average colors per graph: {total_color_count / len(tuning_graphs):.2f}")
    print(f"{'='*70}\n")
    
    # Store only JSON-serializable data in trial user attributes
    trial.set_user_attr('graph_results', graph_results)
    
    # Store graph data in global dict for callback access
    _trial_graph_viz_data[trial.number] = graph_data_for_viz
    
    return total_color_count
