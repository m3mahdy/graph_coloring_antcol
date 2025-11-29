"""
ACO Graph Coloring (Constructive Approach with Parallel Ants)

 ACO Implementation:
- Initialize pheromone trails (tau) on node-color pairs
- Each ant constructs a solution greedily and constructively:
  * Starts from a random node
  * For each uncolored node, selects color based on:
    - Pheromone level (tau): higher pheromone = more attractive
    - Heuristic (eta): fewer expected conflicts = more attractive
  * Builds complete valid solution (no conflicts expected)
- After all ants finish:
  * Evaporate pheromones globally
  * Reinforce pheromones on best solution (min colors)
- Track global best solution across all iterations
- Parallel execution: each ant runs in separate thread

Objective: Minimize number of colors used (with zero conflicts)

Dependencies:
- networkx
- numpy
"""

import networkx as nx
import numpy as np
import random
import threading
from queue import Queue
from copy import deepcopy


class ACOGraphColoring:
    """
    Ant Colony Optimization for Graph Coloring using constructive approach.
    
    Algorithm:
    1. Initialize pheromone matrix (nodes × colors)
    2. Each ant constructs solution greedily:
       - Visit nodes in order
       - Choose color based on pheromone and heuristic
       - Heuristic: minimize expected conflicts and prefer fewer colors
    3. Evaporate pheromones
    4. Reinforce pheromones on iteration-best solution
    5. Track global best (minimum colors, zero conflicts)
    """

    def __init__(self, graph, iterations=30, alpha=1.0, beta=2.0, rho=0.1, ant_count=10, Q=1.0, verbose=False):
        """
        Initialize ACO Graph Coloring algorithm.
        
        Args:
            graph: NetworkX Graph object
            iterations: Number of iterations to run (default: 30)
            alpha: Pheromone importance (default: 1.0)
            beta: Heuristic importance (default: 2.0)
            rho: Evaporation rate (default: 0.1)
            ant_count: Number of ants (default: 10)
            Q: Pheromone deposit intensity (default: 1.0)
            verbose: Print progress information (default: False)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        
        # Estimate initial number of colors (upper bound)
        # Use chromatic number lower bound heuristics
        if self.N < 20:
            self.max_colors = max(3, self.N // 2)
        else:
            self.max_colors = max(10, self.N // 2)
        
        self.initial_num_colors = self.max_colors

        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ant_count = ant_count
        self.Q = Q
        self.verbose = verbose

        # Create node index mapping
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        
        # Initialize pheromone matrix: pheromone[node][color]
        # Start with uniform pheromone levels
        self.pheromone = np.ones((self.N, self.max_colors), dtype=float)
        
        # Build adjacency list for faster neighbor lookup
        self.adjacency = {node: set(self.graph.neighbors(node)) for node in self.nodes}

    def _ant_construct_solution(self, start_node):
        """
        Single ant constructs a complete coloring solution constructively.
        
        Process:
        1. Start from start_node
        2. Visit nodes in sequence (BFS-like or random order)
        3. For each node, choose color probabilistically based on:
           - Pheromone level (tau^alpha)
           - Heuristic value (eta^beta)
        4. ONLY assign valid colors (no conflicts) - conflict-free guaranteed
        
        Args:
            start_node: Starting node for this ant
            
        Returns:
            Dictionary mapping nodes to colors
        """
        solution = {}
        used_colors = set()
        
        # Determine node visitation order (start from start_node)
        nodes_order = self.nodes.copy()
        if start_node in nodes_order:
            nodes_order.remove(start_node)
            nodes_order.insert(0, start_node)
        random.shuffle(nodes_order[1:])  # Shuffle remaining nodes for diversity
        
        # Constructively assign colors to each node
        for node in nodes_order:
            node_idx = self.node_index[node]
            
            # Find valid colors (no conflicts with neighbors)
            valid_colors = []
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            
            for color in range(self.max_colors):
                if color not in neighbor_colors:  # Valid color (no conflict)
                    valid_colors.append(color)
            
            # If no valid colors found, this should not happen with sufficient max_colors
            if not valid_colors:
                # Fallback: add a new color (expand if needed)
                valid_colors = [self.max_colors]
            
            # Calculate scores ONLY for valid colors
            scores = []
            for color in valid_colors:
                # Pheromone component
                tau = self.pheromone[node_idx][color] if color < self.max_colors else 1.0
                tau = tau ** self.alpha
                
                # Heuristic component: prefer already used colors
                if color in used_colors:
                    eta = 2.0  # Higher preference for existing colors
                else:
                    eta = 1.0  # Lower preference for new colors
                eta = eta ** self.beta
                
                scores.append(tau * eta)
            
            # Normalize to probabilities
            scores = np.array(scores)
            score_sum = scores.sum()
            if score_sum > 0:
                probabilities = scores / score_sum
            else:
                # Fallback: uniform distribution over valid colors
                probabilities = np.ones(len(valid_colors)) / len(valid_colors)
            
            # Select color probabilistically from VALID colors only
            chosen_idx = np.random.choice(len(valid_colors), p=probabilities)
            chosen_color = valid_colors[chosen_idx]
            
            solution[node] = int(chosen_color)
            used_colors.add(chosen_color)
        
        return solution

    def _evaluate_solution(self, solution):
        """
        Evaluate a coloring solution.
        
        Returns:
            Tuple of (num_colors_used, conflict_count)
        """
        num_colors = len(set(solution.values()))
        
        conflicts = 0
        for u, v in self.graph.edges():
            if solution[u] == solution[v]:
                conflicts += 1
        
        return num_colors, conflicts

    def _ant_worker_thread(self, start_node, results_queue):
        """
        Worker function for ant thread.
        
        Args:
            start_node: Starting node for this ant
            results_queue: Queue to put results
        """
        solution = self._ant_construct_solution(start_node)
        num_colors, conflicts = self._evaluate_solution(solution)
        
        results_queue.put({
            'solution': solution,
            'num_colors': num_colors,
            'conflicts': conflicts
        })

    def _evaporate_pheromones(self):
        """
        Evaporate pheromones globally by factor (1 - rho).
        """
        self.pheromone *= (1.0 - self.rho)

    def _reinforce_pheromones(self, solution, num_colors):
        """
        Deposit pheromones on the given solution.
        
        Deposit amount inversely proportional to number of colors used.
        
        Args:
            solution: Solution dict (node -> color)
            num_colors: Number of colors in solution
        """
        deposit = self.Q / num_colors
        
        for node, color in solution.items():
            node_idx = self.node_index[node]
            # Only reinforce if color is within pheromone matrix bounds
            if color < self.max_colors:
                self.pheromone[node_idx][color] += deposit

    def run(self):
        """
        Run the ACO algorithm.
        
        Returns:
            Dictionary with:
                - color_count: Number of colors in best solution
                - iterations: Number of iterations executed
        """
        best_global_solution = None
        best_global_colors = float('inf')
        best_global_conflicts = float('inf')
        
        for iteration in range(1, self.iterations + 1):
            # Create ants and run in parallel threads
            results_queue = Queue()
            threads = []
            
            # Assign starting nodes to ants
            if self.ant_count <= self.N:
                start_nodes = random.sample(self.nodes, self.ant_count)
            else:
                start_nodes = [random.choice(self.nodes) for _ in range(self.ant_count)]
            
            # Launch ant threads
            for start_node in start_nodes:
                thread = threading.Thread(
                    target=self._ant_worker_thread,
                    args=(start_node, results_queue)
                )
                thread.start()
                threads.append(thread)
            
            # Wait for all ants to finish
            for thread in threads:
                thread.join()
            
            # Collect results from all ants
            iteration_solutions = []
            while not results_queue.empty():
                iteration_solutions.append(results_queue.get())
            
            # Filter to only conflict-free solutions
            valid_solutions = [sol for sol in iteration_solutions if sol['conflicts'] == 0]
            
            # If no valid solutions found, something went wrong
            if not valid_solutions:
                if self.verbose:
                    print(f"WARNING: Iteration {iteration} produced no valid solutions!")
                # Use best available (even with conflicts) but don't update global best
                iteration_solutions.sort(key=lambda x: (x['conflicts'], x['num_colors']))
                iter_best = iteration_solutions[0]
            else:
                # Find iteration-best among valid solutions (minimize colors)
                valid_solutions.sort(key=lambda x: x['num_colors'])
                iter_best = valid_solutions[0]
                
                # Update global best if better valid solution found
                if iter_best['num_colors'] < best_global_colors:
                    best_global_solution = deepcopy(iter_best['solution'])
                    best_global_colors = iter_best['num_colors']
                    best_global_conflicts = 0  # Always 0 for valid solutions
            
            # Pheromone update (only on valid solutions)
            self._evaporate_pheromones()
            if iter_best['conflicts'] == 0:
                self._reinforce_pheromones(iter_best['solution'], iter_best['num_colors'])
            
            # Verbose output
            if self.verbose:
                print(f"Iteration {iteration}: "
                      f"iter_best={iter_best['num_colors']} colors ({iter_best['conflicts']} conflicts), "
                      f"global_best={best_global_colors} colors ({best_global_conflicts} conflicts)")
            
            # Early stopping if we have found any valid solution
            # (since constructive approach should always produce valid solutions)
            if best_global_conflicts == 0 and best_global_solution is not None:
                # Continue searching for better (fewer colors) solution
                # No early stop - let all iterations complete to find minimum colors
                pass
        
        return {
            'color_count': best_global_colors,
            'iterations': iteration
        }

