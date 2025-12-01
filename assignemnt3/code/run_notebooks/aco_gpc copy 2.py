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
from pathlib import Path


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

    def __init__(self, graph, iterations=30, alpha=1.0, beta=2.0, rho=0.1, ant_count=10, Q=1.0, verbose=False, patience=0.5, trial_number=None, graph_name=None, graph_index=None, tabu_best=None, elitist_ants=0.2):
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
            patience: Number of iterations without improvement before early stopping (None = no early stopping)
            tabu_best: Best known color count from tabu search (for comparison)
            elitist_ants: Fraction of ants that use greedy (deterministic) selection (default: 0.2)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        
        # Estimate initial number of colors (upper bound)
        # Use chromatic number lower bound heuristics
        if self.N < 20:
            self.max_colors = min(100, self.N // 2)
        else:
            self.max_colors = min(1000, self.N // 2)
        
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ant_count = ant_count
        self.Q = Q
        self.verbose = verbose
        self.trial_number = trial_number
        self.graph_name = graph_name
        self.graph_index = graph_index
        self.tabu_best = tabu_best
        self.elitist_ants = int(ant_count * elitist_ants)  # Number of greedy ants
        
        if patience is None:
            self.patience = iterations  # No early stopping
        else:
            self.patience = int(iterations *  patience)

        # Create node index mapping
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        
        # Initialize pheromone matrix: pheromone[node][color]
        # Start with uniform pheromone levels
        self.pheromone = np.ones((self.N, self.max_colors), dtype=float)
        self.min_pheromone = 1.0  # Track minimum pheromone value for matrix expansion
        
        # Build adjacency list for faster neighbor lookup
        self.adjacency = {node: set(self.graph.neighbors(node)) for node in self.nodes}
        
        # Precompute node degrees for ordering heuristic
        self.node_degrees = {node: len(self.adjacency[node]) for node in self.nodes}
        
        # Build adjacency matrix for vectorized operations (performance optimization)
        self.adj_matrix = nx.to_numpy_array(graph, nodelist=self.nodes, dtype=np.int8)
        
        # Seed pheromones with greedy solution for faster convergence
        self._seed_pheromones_with_greedy()
    
    def _seed_pheromones_with_greedy(self):
        """
        Seed pheromone matrix with a greedy DSatur solution for faster convergence.
        Uses proper DSatur ordering for better initial coloring.
        """
        solution = {}
        used_colors = set()
        
        # DSatur: always color the node with highest saturation degree next
        while len(solution) < self.N:
            # Find uncolored node with highest saturation
            best_node = None
            best_saturation = -1
            best_degree = -1
            
            for node in self.nodes:
                if node in solution:
                    continue
                
                # Calculate saturation (distinct colors in neighbors)
                neighbor_colors = {solution[neighbor] for neighbor in self.adjacency[node] if neighbor in solution}
                saturation = len(neighbor_colors)
                degree = self.node_degrees[node]
                
                # Select node with highest saturation, break ties with degree
                if saturation > best_saturation or (saturation == best_saturation and degree > best_degree):
                    best_node = node
                    best_saturation = saturation
                    best_degree = degree
            
            # Color the selected node with first available color
            node = best_node
            neighbor_colors = {solution[neighbor] for neighbor in self.adjacency[node] if neighbor in solution}
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            solution[node] = color
            used_colors.add(color)
        
        # Reinforce pheromones on greedy solution
        num_colors = len(used_colors)
        deposit = self.Q * 2.0 / num_colors  # 2x boost for initial seeding
        
        for node, color in solution.items():
            node_idx = self.node_index[node]
            if color < self.max_colors:  # Only if color is within current range
                self.pheromone[node_idx][color] += deposit

    def _get_node_ordering(self, start_node, solution):
        """
        Get node ordering using DSatur-like strategy:
        - Prioritize high-degree nodes first (Welsh-Powell)
        - Consider saturation degree (number of distinct colors in neighbors)
        
        Args:
            start_node: Starting node
            solution: Current partial solution
            
        Returns:
            List of nodes in order to be colored
        """
        uncolored = [n for n in self.nodes if n not in solution]
        
        # Pre-calculate saturations for all uncolored nodes (more efficient)
        saturations = []
        for node in uncolored:
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            saturations.append((-len(neighbor_colors), -self.node_degrees[node], random.random(), node))
        
        # Sort and extract nodes
        saturations.sort()
        return [item[3] for item in saturations]
    
    def _ant_construct_solution(self, start_node):
        """
        Single ant constructs a complete coloring solution constructively.
        
        Process:
        1. Start from start_node (highest degree preferred)
        2. Visit nodes using DSatur-like ordering:
           - Prioritize nodes with highest saturation degree
           - Break ties with highest degree
        3. For each node, choose color probabilistically based on:
           - Pheromone level (tau^alpha)
           - Heuristic value (eta^beta) - considers saturation and conflicts
        4. ONLY assign valid colors (no conflicts) - conflict-free guaranteed
        
        Args:
            start_node: Starting node for this ant
            
        Returns:
            Dictionary mapping nodes to colors
        """
        solution = {}
        used_colors = set()
        
        # Start with the start_node (high degree)
        current_node = start_node
        
        # Color nodes iteratively using DSatur ordering
        while len(solution) < self.N:
            # Get next batch of nodes to consider (DSatur ordering)
            if current_node is not None and current_node not in solution:
                node = current_node
                current_node = None  # Only use once
            else:
                # Get dynamically ordered uncolored nodes
                nodes_order = self._get_node_ordering(start_node, solution)
                if not nodes_order:
                    break
                node = nodes_order[0]
            
            node_idx = self.node_index[node]
            
            # Find valid colors (no conflicts with neighbors)
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            
            # Optimize: only check colors in used_colors + one new color
            # This is much faster than checking all max_colors
            max_used_color = max(used_colors) if used_colors else -1
            candidate_colors = list(range(max_used_color + 2))  # +2 to include one new color
            
            # Ensure pheromone matrix is large enough for candidate colors
            if candidate_colors and max(candidate_colors) >= self.max_colors:
                self._expand_pheromone_matrix(max(candidate_colors) + 1)
            
            valid_colors = [c for c in candidate_colors if c not in neighbor_colors]
            
            # Ensure we have at least one valid color
            if not valid_colors:
                # Extremely rare case - expand if needed
                new_color = max_used_color + 1
                if new_color >= self.max_colors:
                    self._expand_pheromone_matrix(new_color + 1)
                valid_colors = [new_color]
            
            # Calculate scores ONLY for valid colors with enhanced heuristic (VECTORIZED)
            num_valid = len(valid_colors)
            valid_colors_arr = np.array(valid_colors, dtype=np.int32)
            
            # Pheromone component (vectorized)
            tau_values = self.pheromone[node_idx, valid_colors_arr] ** self.alpha
            
            # Enhanced heuristic component
            # Factor 1: Prefer already used colors (minimize color count)
            color_preference = np.where(np.isin(valid_colors_arr, list(used_colors)), 10.0, 1.0)
            
            # Factor 2: Prefer colors that constrain fewer uncolored neighbors (VECTORIZED)
            # Build solution color array for fast lookup
            solution_colors = np.full(self.N, -1, dtype=np.int32)
            for solved_node, solved_color in solution.items():
                solution_colors[self.node_index[solved_node]] = solved_color
            
            # Calculate constraint score for each valid color
            constraint_scores = np.zeros(num_valid, dtype=np.float32)
            
            # Get uncolored neighbors of current node
            uncolored_neighbor_mask = np.zeros(self.N, dtype=bool)
            for neighbor in self.adjacency[node]:
                if neighbor not in solution:
                    uncolored_neighbor_mask[self.node_index[neighbor]] = True
            
            if uncolored_neighbor_mask.any():
                # For each uncolored neighbor, count how many of ITS neighbors use each color
                uncolored_neighbor_indices = np.where(uncolored_neighbor_mask)[0]
                
                for neighbor_idx in uncolored_neighbor_indices:
                    # Get neighbors of this uncolored neighbor
                    neighbor_neighbors_mask = self.adj_matrix[neighbor_idx] > 0
                    # Get colors used by these neighbors
                    neighbor_neighbor_colors = solution_colors[neighbor_neighbors_mask]
                    # Remove -1 (uncolored)
                    neighbor_neighbor_colors = neighbor_neighbor_colors[neighbor_neighbor_colors >= 0]
                    
                    # Count matches with each valid color
                    for i, color in enumerate(valid_colors_arr):
                        constraint_scores[i] += np.sum(neighbor_neighbor_colors == color)
            
            # Penalize colors that heavily constrain neighbors
            conflict_penalty = 1.0 / (1.0 + constraint_scores * 0.2)
            
            eta_values = color_preference * conflict_penalty
            eta_values = eta_values ** self.beta
            
            scores = tau_values * eta_values
            scores = tau_values * eta_values
            
            # Normalize to probabilities
            score_sum = scores.sum()
            if score_sum > 0:
                probabilities = scores / score_sum
            else:
                # Fallback: uniform distribution over valid colors
                probabilities = np.ones(len(valid_colors)) / len(valid_colors)
            
            # Select color: greedy for elitist ants, probabilistic for others
            # Note: greedy selection handled via high scores, all ants use probabilistic
            chosen_idx = np.random.choice(len(valid_colors), p=probabilities)
            chosen_color = valid_colors[chosen_idx]
            
            solution[node] = int(chosen_color)
            used_colors.add(chosen_color)
        
        return solution

    def _evaluate_solution(self, solution):
        """
        Evaluate a coloring solution.
        
        Returns:
            Number of colors used
        """
        return len(set(solution.values()))

    def _ant_worker_thread(self, start_node, results_queue, ant_id=0):
        """
        Worker function for ant thread.
        
        Args:
            start_node: Starting node for this ant
            results_queue: Queue to put results
            ant_id: Ant identifier for elitist selection
        """
        solution = self._ant_construct_solution(start_node)
        num_colors = self._evaluate_solution(solution)
        
        results_queue.put({
            'solution': solution,
            'num_colors': num_colors
        })

    def _expand_pheromone_matrix(self, new_max_colors):
        """
        Expand pheromone matrix to accommodate more colors.
        New columns are initialized with tracked minimum pheromone value.
        
        Args:
            new_max_colors: New maximum number of colors
        """
        if new_max_colors <= self.max_colors:
            return
        
        # Create new columns initialized with tracked minimum pheromone value
        additional_colors = new_max_colors - self.max_colors
        new_columns = np.full((self.N, additional_colors), self.min_pheromone, dtype=float)
        
        # Append new columns to existing pheromone matrix
        self.pheromone = np.hstack([self.pheromone, new_columns])
        self.max_colors = new_max_colors
    
    def _evaporate_pheromones(self):
        """
        Evaporate pheromones globally by factor (1 - rho).
        Updates minimum pheromone value for new color initialization.
        """
        self.pheromone *= (1.0 - self.rho)
        self.min_pheromone *= (1.0 - self.rho)

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
            self.pheromone[node_idx][color] += deposit


    def run(self):
        """
        Run the ACO algorithm.
        
        Returns:
            Dictionary with:
                - solution: Best solution found (node -> color mapping)
                - color_count: Number of colors in best solution
                - iterations: Number of iterations executed
        """
        self.best_global_solution = None
        best_global_colors = float('inf')
        iterations_without_improvement = 0
        
        for iteration in range(1, self.iterations + 1):
            # Create ants and run
            results_queue = Queue()
            threads = []
            iteration_solutions = []
            
            # Assign starting nodes to ants (prefer high-degree nodes)
            # Sort nodes by degree (highest first) for better starting points
            sorted_nodes = sorted(self.nodes, key=lambda n: self.node_degrees[n], reverse=True)
            
            if self.ant_count <= self.N:
                # If ants <= nodes → assign unique high-degree nodes to ants
                # Use top degree nodes plus some randomization
                top_fraction = max(1, self.ant_count // 2)
                start_nodes = sorted_nodes[:top_fraction]
                # Add random selection from remaining for diversity
                if len(start_nodes) < self.ant_count:
                    remaining = [n for n in sorted_nodes if n not in start_nodes]
                    start_nodes.extend(random.sample(remaining, self.ant_count - len(start_nodes)))
            else:
                # If ants > nodes → ensure full coverage with preference to high-degree
                start_nodes = []
                # 1) Prioritize high-degree nodes
                start_nodes.extend(sorted_nodes)
                # 2) Assign remaining ants to high-degree nodes (weighted)
                remaining_ants = self.ant_count - self.N
                # Weight by degree for selection
                degrees = [self.node_degrees[n] for n in sorted_nodes]
                total_degree = sum(degrees)
                weights = [d / total_degree for d in degrees] if total_degree > 0 else None
                start_nodes.extend(random.choices(sorted_nodes, weights=weights, k=remaining_ants))
            
            # Run ants in parallel
            for ant_id, start_node in enumerate(start_nodes):
                thread = threading.Thread(
                    target=self._ant_worker_thread,
                    args=(start_node, results_queue, ant_id)
                )
                thread.start()
                threads.append(thread)
            
            # Wait for all ants to finish
            for thread in threads:
                thread.join()
            
            # Collect results from all ants
            while not results_queue.empty():
                iteration_solutions.append(results_queue.get())
            
            # Find iteration-best solution (minimize colors)
            iteration_solutions.sort(key=lambda x: x['num_colors'])
            iter_best = iteration_solutions[0]
            
            # Update global best if better solution found
            if iter_best['num_colors'] < best_global_colors:
                self.best_global_solution = deepcopy(iter_best['solution'])
                best_global_colors = iter_best['num_colors']
                iterations_without_improvement = 0  # Reset counter on improvement
            else:
                iterations_without_improvement += 1
            
            # Pheromone update
            self._evaporate_pheromones()
            self._reinforce_pheromones(iter_best['solution'], iter_best['num_colors'])
            
            # Build progress message with available context
            progress_parts = []
            if self.trial_number is not None:
                progress_parts.append(f"Trial {self.trial_number}")
            if self.graph_name is not None:
                progress_parts.append(f"Graph '{self.graph_name}'")
            if self.graph_index is not None:
                progress_parts.append(f"[{self.graph_index}]")
            
            prefix = " | ".join(progress_parts) + " | " if progress_parts else ""
            
            # Build tabu comparison
            tabu_info = ""
            if self.tabu_best is not None:
                diff = best_global_colors - self.tabu_best
                if diff == 0:
                    tabu_info = f" - 🎯 Match Tabu: {self.tabu_best}"
                elif diff > 0:
                    tabu_info = f" - ⚠️ vs Tabu: {self.tabu_best} (+{diff})"
                else:
                    tabu_info = f" - 🏆 Better than Tabu: {self.tabu_best} ({diff})"
            
            # Verbose output - print each iteration on new line
            if iteration % 10 == 0:
                print(f"💬 {prefix}Iteration {iteration}/{self.iterations}: "
                        f"iter_best={iter_best['num_colors']}, "
                        f"global_best={best_global_colors}{tabu_info}")
            
            # Check early stopping condition
            if self.patience is not None and iterations_without_improvement >= self.patience:
                print()  # New line after progress updates
                # make it dangerous
                print(f"❌ {prefix} Early stopping: No improvement for {self.patience} iterations")
                return {
                    'solution': self.best_global_solution,
                    'color_count': best_global_colors,
                    'iterations': iteration
                }
        
        # Print newline after progress updates
        if self.verbose:
            print()
        
        return {
            'solution': self.best_global_solution,
            'color_count': best_global_colors,
            'iterations': iteration
        }

