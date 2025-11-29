"""
Greedy Graph Coloring Algorithm

This module implements a simple greedy coloring algorithm for graph coloring problems.
The algorithm assigns colors to vertices sequentially, choosing the first available color
that doesn't conflict with already-colored neighbors.

Dependencies:
- networkx
- numpy
"""

import networkx as nx
import numpy as np


class GreedyGraphColoring:
    """
    Greedy algorithm for graph coloring.
    
    The algorithm iterates through vertices and assigns the first available color
    that doesn't conflict with neighboring vertices.
    
    CRITICAL: This implementation MUST match Assignment 2 notebook implementation
    (graph_coloring_tabu_search.ipynb) for consistency.
    Core algorithm in _greedy_coloring_from_matrix() matches the notebook's greedy_coloring().
    """
    
    def __init__(self, graph, verbose=False):
        """
        Initialize Greedy Graph Coloring algorithm.
        
        Args:
            graph: NetworkX Graph object
            verbose: Print progress information (default: False)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        self.verbose = verbose
        
    def _adjacency_matrix_from_graph(self):
        """
        Create adjacency matrix from NetworkX graph.
        
        Returns:
            numpy array representing the adjacency matrix
        """
        adj_matrix = np.zeros((self.N, self.N), dtype=int)
        node_index = {node: i for i, node in enumerate(self.nodes)}
        
        for u, v in self.graph.edges():
            i, j = node_index[u], node_index[v]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
            
        return adj_matrix
    
    def _greedy_coloring_from_matrix(self, adj_matrix):
        """
        Apply greedy coloring to adjacency matrix.
        
        CRITICAL: Algorithm matches Assignment 2 notebook (greedy_coloring function).
        Differences from Assignment 2:
        - Returns tuple (color_dict, color_count) instead of just color_dict
        - Uses 1-based coloring (colors start from 1, not 0)
        
        Args:
            adj_matrix: Adjacency matrix representing the graph
            
        Returns:
            Tuple of (color_dict, color_count) where color_dict maps node indices to colors
        """
        n = len(adj_matrix)
        colors = [0] * n
        colors[0] = 1
        
        for i in range(1, n):
            available_colors = set(range(1, n + 1))
            
            for j in range(n):
                if adj_matrix[i][j] == 1:
                    if colors[j] in available_colors:
                        available_colors.remove(colors[j])
            
            colors[i] = min(available_colors)
        
        color_dict = {i: colors[i] for i in range(n)}
        color_count = len(set(color_dict.values()))
        
        return color_dict, color_count
    
    def _evaluate_solution(self, solution):
        """
        Evaluate a coloring solution by counting conflicts.
        
        Args:
            solution: Dictionary mapping node labels to colors
            
        Returns:
            Tuple of (color_count, conflict_count)
        """
        color_count = len(set(solution.values()))
        conflicts = 0
        
        for u, v in self.graph.edges():
            if solution[u] == solution[v]:
                conflicts += 1
                
        return color_count, conflicts
    
    def run(self):
        """
        Run the Greedy coloring algorithm.
        
        Returns:
            Dictionary with best_solution, color_count, conflict_count, and iterations
        """
        adj_matrix = self._adjacency_matrix_from_graph()
        color_dict_indices, color_count = self._greedy_coloring_from_matrix(adj_matrix)
        
        solution = {self.nodes[i]: color_dict_indices[i] for i in range(self.N)}
        
        color_count_eval, conflict_count = self._evaluate_solution(solution)
        
        if self.verbose:
            print(f"Greedy: colors={color_count_eval}, conflicts={conflict_count}")
        
        return {
            "best_solution": solution,
            "color_count": color_count_eval,
            "conflict_count": conflict_count,
            "iterations": 1
        }
