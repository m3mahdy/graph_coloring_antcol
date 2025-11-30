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

    def __init__(self, graph, iterations=30, alpha=1.0, beta=2.0, rho=0.1, ant_count=10, Q=1.0, verbose=False, viz_dir=None, patience=0.5):
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
            viz_dir: Directory to save visualizations (None = no visualization)
            patience: Number of iterations without improvement before early stopping (None = no early stopping)
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
        self.viz_dir = viz_dir
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
        
        # Initialize visualizer if visualization is enabled (after pheromone is set up)
        self.visualizer = None
        if self.viz_dir:
            from aco_visualization import ACOVisualizer
            self.visualizer = ACOVisualizer(self)

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
            
            # If no valid colors found, expand pheromone matrix
            if not valid_colors:
                # Need to add a new color - expand matrix first
                self._expand_pheromone_matrix(self.max_colors + 1)
                valid_colors = [self.max_colors - 1]  # Use the newly added color
            
            # Calculate scores ONLY for valid colors
            scores = []
            for color in valid_colors:
                # Pheromone component
                tau = self.pheromone[node_idx][color]
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
                # Why ? Because all scores are zero, likely due to very low pheromone levels.
                # if so, this will cause division by zero. So we assign equal probabilities.
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
            Number of colors used
        """
        return len(set(solution.values()))

    def _ant_worker_thread(self, start_node, results_queue):
        """
        Worker function for ant thread.
        
        Args:
            start_node: Starting node for this ant
            results_queue: Queue to put results
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

    def _ant_construct_solution_with_viz(self, start_node, ant_id, iteration, save_dir):
        """
        Construct solution with step-by-step visualization.
        
        Args:
            start_node: Starting node for this ant
            ant_id: Ant identifier
            iteration: Current iteration
            save_dir: Directory to save step images
            
        Returns:
            Dictionary mapping nodes to colors
        """
        from pathlib import Path
        solution = {}
        used_colors = set()
        
        # Determine node visitation order
        nodes_order = self.nodes.copy()
        if start_node in nodes_order:
            nodes_order.remove(start_node)
            nodes_order.insert(0, start_node)
        random.shuffle(nodes_order[1:])
        
        # Create ant-specific directory
        ant_dir = Path(save_dir) / f'ant_{ant_id}'
        ant_dir.mkdir(parents=True, exist_ok=True)
        
        # Constructively assign colors with visualization
        for step, node in enumerate(nodes_order):
            node_idx = self.node_index[node]
            
            # Find valid colors
            valid_colors = []
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            
            for color in range(self.max_colors):
                if color not in neighbor_colors:
                    valid_colors.append(color)
            
            if not valid_colors:
                # Need to add a new color - expand matrix first
                self._expand_pheromone_matrix(self.max_colors + 1)
                valid_colors = [self.max_colors - 1]  # Use the newly added color
            
            # Calculate scores and select color
            scores = []
            for color in valid_colors:
                tau = self.pheromone[node_idx][color]
                tau = tau ** self.alpha
                eta = 2.0 if color in used_colors else 1.0
                eta = eta ** self.beta
                scores.append(tau * eta)
            
            scores = np.array(scores)
            probabilities = scores / scores.sum() if scores.sum() > 0 else np.ones(len(valid_colors)) / len(valid_colors)
            chosen_idx = np.random.choice(len(valid_colors), p=probabilities)
            chosen_color = valid_colors[chosen_idx]
            
            solution[node] = int(chosen_color)
            used_colors.add(chosen_color)
            
            # Visualize current step
            self.visualizer.plot_partial_solution(solution, nodes_order[:step+1], ant_id, iteration, step, 
                                       save_path=ant_dir / f'step_{step:03d}.png')
        
        return solution

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
        
        # Plot initial pheromone state if visualizing
        if self.viz_dir:
            initial_dir = Path(self.viz_dir) / 'iteration_000'
            initial_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer.plot_pheromone_heatmap(0, save_path=initial_dir / 'pheromone.png')
        
        for iteration in range(1, self.iterations + 1):
            # Create iteration directory for visualizations
            if self.viz_dir:
                iter_dir = Path(self.viz_dir) / f'iteration_{iteration:03d}'
                iter_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ants and run (with or without visualization)
            results_queue = Queue()
            threads = []
            iteration_solutions = []
            
            # Assign starting nodes to ants
            if self.ant_count <= self.N:
                # If ants <= nodes → assign unique nodes to ants
                start_nodes = random.sample(self.nodes, self.ant_count)

            else:
                # If ants > nodes → ensure full coverage of all nodes
                start_nodes = []

                # 1) Guarantee that each node is assigned to at least one ant
                start_nodes.extend(self.nodes)

                # 2) Assign the remaining ants to random nodes (with replacement)
                remaining_ants = self.ant_count - self.N
                start_nodes.extend(random.choices(self.nodes, k=remaining_ants))

            
            # Run ants with visualization if enabled
            if self.viz_dir:
                # Sequential execution with visualization for each ant
                for ant_id, start_node in enumerate(start_nodes, 1):
                    solution = self._ant_construct_solution_with_viz(start_node, ant_id, iteration, iter_dir)
                    num_colors = self._evaluate_solution(solution)
                    iteration_solutions.append({
                        'solution': solution,
                        'num_colors': num_colors
                    })
                    # Save final ant solution in ant folder
                    ant_dir = iter_dir / f'ant_{ant_id}'
                    self.visualizer.plot_ant_solution(solution, ant_id, iteration, 
                                          save_path=ant_dir / 'final_solution.png')
            else:
                # Parallel execution without visualization (faster)
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
            
            # Verbose output - update same line
            print(f"\rIteration {iteration}/{self.iterations}: "
                    f"iter_best={iter_best['num_colors']} colors, "
                    f"global_best={best_global_colors} colors", end='', flush=True)
            
            # Check early stopping condition
            if self.patience is not None and iterations_without_improvement >= self.patience:
                print()  # New line after progress updates
                # make it dangerous
                print(f"\n ❌ Early stopping: No improvement for {self.patience} iterations")
                return {
                    'solution': self.best_global_solution,
                    'color_count': best_global_colors,
                    'iterations': iteration
                }
            
            # Save iteration visualizations
            if self.viz_dir:
                # Save pheromone heatmap
                self.visualizer.plot_pheromone_heatmap(iteration, save_path=iter_dir / 'pheromone.png')
                # Save best global solution so far
                if self.best_global_solution:
                    self.visualizer.plot_best_solution(self.best_global_solution, iteration, save_path=iter_dir / 'best_global.png')
        
        # Print newline after progress updates
        if self.verbose:
            print()
        
        # Save final best solution at root of visualization directory
        if self.viz_dir and self.best_global_solution:
            self.visualizer.plot_best_solution(
                self.best_global_solution, 
                iteration, 
                save_path=Path(self.viz_dir) / 'final_best_solution.png'
            )
        
        return {
            'solution': self.best_global_solution,
            'color_count': best_global_colors,
            'iterations': iteration
        }

