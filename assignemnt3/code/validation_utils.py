"""
Validation Utilities for Graph Coloring Solutions

This module provides functions to validate graph coloring solutions,
ensuring they are complete (all nodes colored) and valid (no conflicts).

Responsibilities:
- Count conflicts in a coloring solution
- Validate solution completeness
- Generate validation reports
"""

import networkx as nx


def count_conflicts(graph, solution):
    """
    Count the number of conflicting edges in a coloring solution.
    
    A conflict occurs when two adjacent nodes (connected by an edge)
    have the same color assigned.
    
    Args:
        graph: NetworkX Graph object
        solution: Dictionary mapping nodes to colors {node: color}
        
    Returns:
        int: Number of conflicting edges (should be 0 for valid coloring)
    """
    conflicts = 0
    for u, v in graph.edges():
        if u in solution and v in solution:
            if solution[u] == solution[v]:
                conflicts += 1
    return conflicts


def validate_solution(graph, solution, verbose=False):
    """
    Validate a graph coloring solution for completeness and correctness.
    
    Checks:
    1. All nodes are colored (solution is complete)
    2. No adjacent nodes have the same color (no conflicts)
    
    Args:
        graph: NetworkX Graph object
        solution: Dictionary mapping nodes to colors {node: color}
        verbose: Print validation details (default: False)
        
    Returns:
        dict: Validation results with keys:
            - 'valid': True if solution is valid and complete
            - 'complete': True if all nodes are colored
            - 'conflict_free': True if no conflicts exist
            - 'num_colors': Number of colors used
            - 'conflicts': Number of conflicting edges
            - 'missing_nodes': List of nodes not in solution
            - 'extra_nodes': List of nodes in solution but not in graph
    """
    graph_nodes = set(graph.nodes())
    solution_nodes = set(solution.keys())
    
    # Check completeness
    missing_nodes = list(graph_nodes - solution_nodes)
    extra_nodes = list(solution_nodes - graph_nodes)
    complete = len(missing_nodes) == 0 and len(extra_nodes) == 0
    
    # Check conflicts
    conflicts = count_conflicts(graph, solution)
    conflict_free = conflicts == 0
    
    # Count colors
    num_colors = len(set(solution.values())) if solution else 0
    
    # Overall validity
    valid = complete and conflict_free
    
    result = {
        'valid': valid,
        'complete': complete,
        'conflict_free': conflict_free,
        'num_colors': num_colors,
        'conflicts': conflicts,
        'missing_nodes': missing_nodes,
        'extra_nodes': extra_nodes
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SOLUTION VALIDATION")
        print("="*60)
        print(f"Graph: {len(graph_nodes)} nodes, {len(graph.edges())} edges")
        print(f"Solution: {len(solution_nodes)} nodes colored")
        print(f"\nCompleteness:")
        print(f"  All nodes colored: {'✓' if complete else '✗'}")
        if missing_nodes:
            print(f"  Missing nodes ({len(missing_nodes)}): {missing_nodes[:10]}{'...' if len(missing_nodes) > 10 else ''}")
        if extra_nodes:
            print(f"  Extra nodes ({len(extra_nodes)}): {extra_nodes[:10]}{'...' if len(extra_nodes) > 10 else ''}")
        
        print(f"\nCorrectness:")
        print(f"  Conflict-free: {'✓' if conflict_free else '✗'}")
        print(f"  Conflicts: {conflicts}")
        
        print(f"\nColors:")
        print(f"  Number of colors used: {num_colors}")
        
        print(f"\nOverall:")
        print(f"  Valid solution: {'✅ YES' if valid else '❌ NO'}")
        print("="*60)
    
    return result


def find_conflicting_edges(graph, solution):
    """
    Find all conflicting edges in a coloring solution.
    
    Args:
        graph: NetworkX Graph object
        solution: Dictionary mapping nodes to colors {node: color}
        
    Returns:
        list: List of conflicting edges as tuples [(u, v, color), ...]
              where u and v are adjacent nodes with the same color
    """
    conflicting_edges = []
    for u, v in graph.edges():
        if u in solution and v in solution:
            if solution[u] == solution[v]:
                conflicting_edges.append((u, v, solution[u]))
    return conflicting_edges


def get_color_distribution(solution):
    """
    Get the distribution of colors in a solution.
    
    Args:
        solution: Dictionary mapping nodes to colors {node: color}
        
    Returns:
        dict: Dictionary mapping color to number of nodes {color: count}
    """
    color_counts = {}
    for color in solution.values():
        color_counts[color] = color_counts.get(color, 0) + 1
    return color_counts


def validate_multiple_solutions(graph, solutions, algorithm_names=None):
    """
    Validate multiple solutions for the same graph.
    
    Args:
        graph: NetworkX Graph object
        solutions: List of solution dictionaries
        algorithm_names: Optional list of algorithm names for each solution
        
    Returns:
        list: List of validation results for each solution
    """
    if algorithm_names is None:
        algorithm_names = [f"Solution {i+1}" for i in range(len(solutions))]
    
    results = []
    print("\n" + "="*70)
    print("MULTIPLE SOLUTION VALIDATION")
    print("="*70)
    
    for name, solution in zip(algorithm_names, solutions):
        result = validate_solution(graph, solution, verbose=False)
        result['algorithm'] = name
        results.append(result)
        
        status = '✅' if result['valid'] else '❌'
        print(f"{status} {name}: {result['num_colors']} colors, "
              f"{result['conflicts']} conflicts, "
              f"{'complete' if result['complete'] else 'incomplete'}")
    
    print("="*70)
    return results
