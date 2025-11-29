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
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

    def __init__(self, graph, iterations=30, alpha=1.0, beta=2.0, rho=0.1, ant_count=10, Q=1.0, verbose=False, viz_dir=None):
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
        self.viz_dir = viz_dir

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
                valid_colors = [self.max_colors]
            
            # Calculate scores and select color
            scores = []
            for color in valid_colors:
                tau = self.pheromone[node_idx][color] if color < self.max_colors else 1.0
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
            self._plot_partial_solution(solution, nodes_order[:step+1], ant_id, iteration, step, 
                                       save_path=ant_dir / f'step_{step:03d}.png')
        
        return solution
    
    def _plot_partial_solution(self, solution, colored_nodes, ant_id, iteration, step, save_path=None):
        """
        Plot partial solution showing which nodes are colored so far.
        
        Args:
            solution: Current partial solution
            colored_nodes: List of nodes colored so far
            ant_id: Ant identifier
            iteration: Current iteration
            step: Current step number
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create color map
        num_colors = len(set(solution.values())) if solution else 1
        color_map = cm.get_cmap('tab20', max(num_colors, 3))
        
        # Separate colored and uncolored nodes
        colored_node_list = [n for n in self.graph.nodes() if n in solution]
        uncolored_node_list = [n for n in self.graph.nodes() if n not in solution]
        current_node = colored_nodes[-1] if colored_nodes else None
        
        # Create node colors for colored nodes
        colored_node_colors = [color_map(solution[node] % 20) for node in colored_node_list]
        
        # Draw graph
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=50)
        
        # Draw edges first
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1.5, ax=ax)
        
        # Draw uncolored nodes as empty circles (white with black border)
        if uncolored_node_list:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=uncolored_node_list,
                                  node_color='white', node_size=300, 
                                  edgecolors='black', linewidths=2, ax=ax)
        
        # Draw colored nodes (excluding current node)
        colored_except_current = [n for n in colored_node_list if n != current_node]
        if colored_except_current:
            colored_except_current_colors = [color_map(solution[node] % 20) for node in colored_except_current]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=colored_except_current,
                                  node_color=colored_except_current_colors, node_size=300, 
                                  edgecolors='black', linewidths=1.5, ax=ax)
        
        # Draw current node being colored with red circle border
        if current_node is not None:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[current_node],
                                  node_color=[color_map(solution[current_node] % 20)],
                                  node_size=300, edgecolors='red', linewidths=4, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'Ant {ant_id} - Iteration {iteration} - Step {step+1}/{len(self.nodes)}\n'
                     f'{len(colored_nodes)} nodes colored, {num_colors} colors used', 
                     fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_pheromone_heatmap(self, iteration, save_path=None):
        """
        Plot pheromone trails as a heatmap (nodes × colors) with values.
        
        Args:
            iteration: Current iteration number (for title)
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap with light blue gradient colormap
        im = ax.imshow(self.pheromone, cmap='Blues', aspect='auto', interpolation='nearest',
                      vmin=self.pheromone.min(), vmax=self.pheromone.max())
        
        # Add text annotations with ALL pheromone values
        pheromone_max = self.pheromone.max()
        pheromone_min = self.pheromone.min()
        # Use higher threshold (70%) so more cells have dark text on light background
        threshold = pheromone_min + (pheromone_max - pheromone_min) * 0.7
        
        for i in range(self.N):
            for j in range(self.max_colors):
                value = self.pheromone[i, j]
                # Choose text color based on background brightness
                text_color = 'white' if value > threshold else 'darkblue'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=10 if self.N > 20 else 12, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Color Index', fontsize=12)
        ax.set_ylabel('Node Index', fontsize=12)
        ax.set_title(f'Pheromone Trails - Iteration {iteration}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pheromone Level', fontsize=11)
        
        # Set ticks
        if self.N <= 50:
            ax.set_yticks(range(0, self.N, max(1, self.N // 10)))
        if self.max_colors <= 50:
            ax.set_xticks(range(0, self.max_colors, max(1, self.max_colors // 10)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_ant_solution(self, solution, ant_id, iteration, save_path=None):
        """
        Plot the graph with nodes colored according to an ant's solution.
        
        Args:
            solution: Dictionary mapping nodes to colors
            ant_id: Ant identifier (for title)
            iteration: Current iteration number (for title)
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get unique colors and create color map
        num_colors = len(set(solution.values()))
        color_map = cm.get_cmap('tab20', num_colors)
        
        # Map each node to a matplotlib color
        node_colors = [color_map(solution[node] % 20) for node in self.graph.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=50)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                               node_size=300, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1.5, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Title with color count
        ax.set_title(f'Ant {ant_id} Solution - Iteration {iteration}\n{num_colors} colors used', 
                     fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_best_solution(self, iteration, save_path=None):
        """
        Plot the current best global solution.
        
        Args:
            iteration: Current iteration number (for title)
            save_path: Optional path to save the figure
        """
        if self.best_global_solution is None:
            print("No solution available to plot.")
            return
        
        self.plot_ant_solution(self.best_global_solution, "Best", iteration, save_path)

    def run(self):
        """
        Run the ACO algorithm.
        
        Returns:
            Dictionary with:
                - color_count: Number of colors in best solution
                - iterations: Number of iterations executed
        """
        self.best_global_solution = None
        best_global_colors = float('inf')
        best_global_conflicts = float('inf')
        
        # Plot initial pheromone state if visualizing
        if self.viz_dir:
            from pathlib import Path
            initial_dir = Path(self.viz_dir) / 'iteration_000'
            initial_dir.mkdir(parents=True, exist_ok=True)
            self.plot_pheromone_heatmap(0, save_path=initial_dir / 'pheromone.png')
        
        for iteration in range(1, self.iterations + 1):
            # Create iteration directory for visualizations
            if self.viz_dir:
                from pathlib import Path
                iter_dir = Path(self.viz_dir) / f'iteration_{iteration:03d}'
                iter_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ants and run (with or without visualization)
            results_queue = Queue()
            threads = []
            iteration_solutions = []
            
            # Assign starting nodes to ants
            if self.ant_count <= self.N:
                start_nodes = random.sample(self.nodes, self.ant_count)
            else:
                start_nodes = [random.choice(self.nodes) for _ in range(self.ant_count)]
            
            # Run ants with visualization if enabled
            if self.viz_dir:
                # Sequential execution with visualization for each ant
                for ant_id, start_node in enumerate(start_nodes, 1):
                    solution = self._ant_construct_solution_with_viz(start_node, ant_id, iteration, iter_dir)
                    num_colors, conflicts = self._evaluate_solution(solution)
                    iteration_solutions.append({
                        'solution': solution,
                        'num_colors': num_colors,
                        'conflicts': conflicts
                    })
                    # Save final ant solution in ant folder
                    ant_dir = iter_dir / f'ant_{ant_id}'
                    self.plot_ant_solution(solution, ant_id, iteration, 
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
                    self.best_global_solution = deepcopy(iter_best['solution'])
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
            
            # Save iteration visualizations
            if self.viz_dir:
                # Save pheromone heatmap
                self.plot_pheromone_heatmap(iteration, save_path=iter_dir / 'pheromone.png')
                # Save best global solution so far
                if self.best_global_solution:
                    self.plot_best_solution(iteration, save_path=iter_dir / 'best_global.png')
            
            # Early stopping if we have found any valid solution
            # (since constructive approach should always produce valid solutions)
            if best_global_conflicts == 0 and self.best_global_solution is not None:
                # Continue searching for better (fewer colors) solution
                # No early stop - let all iterations complete to find minimum colors
                pass
        
        return {
            'color_count': best_global_colors,
            'iterations': iteration
        }

