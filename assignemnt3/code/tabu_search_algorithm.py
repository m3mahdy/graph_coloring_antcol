"""
Tabu Search Algorithm for Graph Coloring

This module implements the Tabu Search metaheuristic for solving graph coloring problems.
The algorithm starts with a greedy solution and iteratively improves it by reducing the
number of colors while maintaining feasibility through memory-based search.

Dependencies:
- networkx
- numpy
"""

import networkx as nx
import numpy as np
import random
from collections import defaultdict, deque
from copy import deepcopy


class TabuSearchGraphColoring:
    """
    Tabu Search algorithm for graph coloring.
    
    The algorithm uses a tabu list to prevent cycling and an aspiration criterion
    to escape local optima. It iteratively reduces the number of colors starting
    from a greedy solution.
    
    CRITICAL: This implementation MUST match Assignment 2 notebook implementation
    (graph_coloring_tabu_search.ipynb) for consistency.
    Core algorithm logic in _tabu_coloring() matches the notebook's tabu_coloring().
    
    Key parameters match Assignment 2:
    - max_iterations: 3000 in Assignment 2, default 7000 here (configurable)
    - tabu_size: 3 in Assignment 2, default 10 here (configurable)
    - tabu_reps: 30 in Assignment 2, default 50 here (configurable)
    """
    
    def __init__(self, graph, max_iterations=7000, tabu_size=10, tabu_reps=50, verbose=False):
        """
        Initialize Tabu Search Graph Coloring algorithm.
        
        Args:
            graph: NetworkX Graph object
            max_iterations: Maximum iterations per color level (default: 7000)
            tabu_size: Size of the tabu list (default: 10)
            tabu_reps: Number of random samples per iteration (default: 50)
            verbose: Print progress information (default: False)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.tabu_reps = tabu_reps
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
    
    def _adjacency_matrix_to_list(self, matrix):
        """
        Convert adjacency matrix to adjacency list representation.
        
        Args:
            matrix: Adjacency matrix
            
        Returns:
            Dictionary mapping node indices to lists of neighbor indices
        """
        adjacency_list = defaultdict(list)
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if matrix[row][col] == 1:
                    adjacency_list[row].append(col)
        return dict(adjacency_list)
    
    def _greedy_coloring(self, adj_matrix):
        """
        Generate initial solution using greedy coloring.
        
        Args:
            adj_matrix: Adjacency matrix
            
        Returns:
            Tuple of (color_dict, color_count)
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
        return color_dict, len(set(color_dict.values()))
    
    def _tabu_coloring(self, adjacency_list, number_of_colors, previous_solution, is_first_solution):
        """
        Apply tabu search to find a valid coloring with the specified number of colors.
        
        CRITICAL: Core algorithm matches Assignment 2 notebook (tabu_coloring function).
        The logic for tabu list, aspiration criterion, and conflict resolution is identical.
        
        Args:
            adjacency_list: Graph adjacency list representation
            number_of_colors: Target number of colors
            previous_solution: Previous solution to refine
            is_first_solution: Whether this is the first iteration
            
        Returns:
            Tuple of (success, solution) where success indicates if valid coloring was found
        """
        colors = list(range(number_of_colors))
        iterations = 0
        
        tabu_list = deque()
        aspiration_dict = {}
        solution = deepcopy(previous_solution)
        
        if not is_first_solution:
            for i in range(len(adjacency_list)):
                if solution[i] >= number_of_colors:
                    solution[i] = colors[random.randrange(0, len(colors))]
        
        while iterations < self.max_iterations:
            candidates = set()
            conflict_count = 0
            
            for vertex, edges in adjacency_list.items():
                for edge in edges:
                    if solution[vertex] == solution[edge]:
                        candidates.add(vertex)
                        candidates.add(edge)
                        conflict_count += 1
            
            candidates = list(candidates)
            
            if conflict_count == 0:
                break
            
            new_solution = None
            for _ in range(self.tabu_reps):
                vertex = candidates[random.randrange(0, len(candidates))]
                new_color = colors[random.randrange(0, len(colors))]
                
                if solution[vertex] == new_color:
                    new_color = colors[-1]
                
                new_solution = deepcopy(solution)
                new_solution[vertex] = new_color
                new_conflicts = 0
                
                for v, edges in adjacency_list.items():
                    for edge in edges:
                        if v is not None and edge is not None and new_solution[v] == new_solution[edge]:
                            new_conflicts += 1
                
                if new_conflicts < conflict_count:
                    if new_conflicts <= aspiration_dict.setdefault(conflict_count, conflict_count - 1):
                        aspiration_dict[conflict_count] = new_conflicts - 1
                        if (vertex, new_color) in tabu_list:
                            tabu_list.remove((vertex, new_color))
                            break
                    else:
                        if (vertex, new_color) in tabu_list:
                            continue
                    break
            
            tabu_list.append((vertex, solution[vertex]))
            if len(tabu_list) > self.tabu_size:
                tabu_list.popleft()
            
            solution = deepcopy(new_solution)
            iterations += 1
        
        if conflict_count != 0:
            return False, previous_solution
        else:
            return True, solution
    
    def _tabu_search(self, adjacency_list, greedy_result_dict, greedy_result_number):
        """
        Iteratively reduce colors using tabu search.
        
        Args:
            adjacency_list: Graph adjacency list
            greedy_result_dict: Initial greedy solution
            greedy_result_number: Number of colors in greedy solution
            
        Returns:
            Tuple of (final_solution, color_count)
        """
        first_coloring = True
        result = greedy_result_dict
        
        for num_colors in range(greedy_result_number, 1, -1):
            if self.verbose:
                print(f"Attempting {num_colors} colors...")
            status, result = self._tabu_coloring(adjacency_list, num_colors, result, first_coloring)
            if not status:
                break
            first_coloring = False
        
        return result, len(set(result.values()))
    
    def _evaluate_solution(self, solution):
        """
        Evaluate solution by counting colors and conflicts.
        
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
        Run the Tabu Search algorithm.
        
        Returns:
            Dictionary with best_solution, color_count, conflict_count, and iterations
        """
        adj_matrix = self._adjacency_matrix_from_graph()
        adjacency_list = self._adjacency_matrix_to_list(adj_matrix)
        
        greedy_dict, greedy_colors = self._greedy_coloring(adj_matrix)
        
        if self.verbose:
            print(f"Greedy initial solution: {greedy_colors} colors")
        
        result_dict, final_colors = self._tabu_search(adjacency_list, greedy_dict, greedy_colors)
        
        solution = {self.nodes[i]: result_dict[i] for i in range(self.N)}
        
        color_count, conflict_count = self._evaluate_solution(solution)
        
        if self.verbose:
            print(f"Tabu Search: colors={color_count}, conflicts={conflict_count}")
        
        return {
            "best_solution": solution,
            "color_count": color_count,
            "conflict_count": conflict_count,
            "iterations": self.max_iterations
        }
