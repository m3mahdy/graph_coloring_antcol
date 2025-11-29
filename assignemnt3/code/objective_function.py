"""
Objective function for ACO hyperparameter tuning with Optuna.
"""


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
        float: Best (minimum) color count across all tuning graphs (to be minimized)
    """
    graph_results = {}
    best_color_count = float('inf')
    
    # Iterate through all graphs in the tuning dataset
    for graph_name, graph in tuning_graphs:
        # Create ACO instance with all suggested hyperparameters
        aco = aco_class(graph=graph, **params)
        
        # Run ACO optimization (uses parameters set during initialization)
        result = aco.run()
        
        # Store results for this graph
        graph_results[graph_name] = {
            'color_count': result['color_count'],
            'conflict_count': result['conflict_count'],
            'iterations_used': result['iterations']
        }
        
        # Track the best color count
        if result['color_count'] < best_color_count:
            best_color_count = result['color_count']
        
        print(f"  [{graph_name}] Colors: {result['color_count']}, Conflicts: {result['conflict_count']}")
    
    # Store graph results in trial user attributes
    trial.set_user_attr('graph_results', graph_results)
    
    return best_color_count
