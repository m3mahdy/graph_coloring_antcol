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
from numba import njit, prange
from aco_warmup import run_warmup


# ============================================================================
# CONSTANTS AND LOOKUPS
# ============================================================================

# Ant type constants (for data consistency across the system)
class AntType:
    """Ant behavior types for ACO algorithm."""
    GREEDY = 'greedy'
    PHEROMONE = 'pheromone'
    ANTI_PHEROMONE = 'anti_pheromone'


# Node ordering strategy constants
class Strategy:
    """Node ordering strategies for construction phase."""
    DSATUR = 0  # Pure DSatur (saturation-first, then degree)
    DEGREE_FIRST = 1  # Degree-first (Welsh-Powell style)
    BALANCED = 2  # Balanced (saturation with controlled randomization)


# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS (JIT-compiled for performance)
# ============================================================================

@njit(cache=True)
def calculate_constraint_scores_numba(valid_colors_arr, solution_colors, adj_matrix, 
                                       uncolored_neighbor_indices):
    """
    Calculate constraint scores for valid colors using Numba JIT compilation.
    
    For each valid color, counts how many times it appears in the neighborhoods
    of uncolored neighbors of the current node.
    
    Args:
        valid_colors_arr: Array of valid color indices
        solution_colors: Array mapping node_idx -> color (-1 if uncolored)
        adj_matrix: Adjacency matrix (N x N)
        uncolored_neighbor_indices: Indices of uncolored neighbors
        
    Returns:
        Array of constraint scores for each valid color
    """
    num_valid = len(valid_colors_arr)
    constraint_scores = np.zeros(num_valid, dtype=np.float32)
    
    # For each uncolored neighbor
    for neighbor_idx in uncolored_neighbor_indices:
        # Get colors used by neighbors of this uncolored neighbor
        for nn_idx in range(len(solution_colors)):
            if adj_matrix[neighbor_idx, nn_idx] > 0:  # If neighbor
                nn_color = solution_colors[nn_idx]
                if nn_color >= 0:  # If colored
                    # Count matches with each valid color
                    for i in range(num_valid):
                        if valid_colors_arr[i] == nn_color:
                            constraint_scores[i] += 1
    
    return constraint_scores


@njit(cache=True)
def calculate_color_scores_numba(pheromone_node, valid_colors_arr, used_colors_set,
                                  constraint_scores, alpha, beta, color_preference_weight,
                                  constraint_penalty_weight):
    """
    Calculate final scores for color selection using Numba JIT compilation.
    
    Combines pheromone levels with heuristic values (color preference and conflict penalty).
    
    Args:
        pheromone_node: Pheromone values for current node (all colors)
        valid_colors_arr: Array of valid color indices
        used_colors_set: Set of already used colors (as array for Numba compatibility)
        constraint_scores: Constraint scores for each valid color
        alpha: Pheromone importance parameter
        beta: Heuristic importance parameter
        color_preference_weight: Weight for used colors preference (default: 10.0)
        constraint_penalty_weight: Weight for constraint penalty (default: 0.2)
        
    Returns:
        Array of final scores for each valid color
    """
    num_valid = len(valid_colors_arr)
    scores = np.zeros(num_valid, dtype=np.float64)
    
    for i in range(num_valid):
        color = valid_colors_arr[i]
        
        # Pheromone component
        tau = pheromone_node[color] ** alpha
        
        # Color preference (configurable weight for used colors, 1x for new colors)
        color_preference = color_preference_weight if color in used_colors_set else 1.0
        
        # Conflict penalty (prefer colors that constrain fewer neighbors)
        conflict_penalty = 1.0 / (1.0 + constraint_scores[i] * constraint_penalty_weight)
        
        # Combined heuristic
        eta = color_preference * conflict_penalty
        eta = eta ** beta
        
        scores[i] = tau * eta
    
    return scores


@njit(cache=True)
def calculate_least_constraining_scores_numba(valid_colors_arr, solution_colors, adj_matrix, 
                                                node_idx, uncolored_indices):
    """
    Calculate how constraining each valid color is for uncolored nodes.
    
    For each valid color, counts how many uncolored nodes would lose this color as an option
    if we assign it to the current node.
    
    Lower score = less constraining = better choice (leaves more options for neighbors)
    
    Args:
        valid_colors_arr: Array of valid color indices
        solution_colors: Array mapping node_idx -> color (-1 if uncolored)
        adj_matrix: Adjacency matrix (N x N)
        node_idx: Index of current node being colored
        uncolored_indices: Indices of all uncolored nodes
        
    Returns:
        Array of constraining scores for each valid color (lower is better)
    """
    num_valid = len(valid_colors_arr)
    constraining_scores = np.zeros(num_valid, dtype=np.float32)
    
    # For each valid color
    for i in range(num_valid):
        color = valid_colors_arr[i]
        constraint_count = 0.0
        
        # Check each uncolored node
        for uncolored_idx in uncolored_indices:
            # If this uncolored node is a neighbor of current node
            if adj_matrix[node_idx, uncolored_idx] > 0:
                # Count how many of its colored neighbors already use this color
                neighbor_uses_color = False
                for neighbor_idx in range(len(solution_colors)):
                    if (adj_matrix[uncolored_idx, neighbor_idx] > 0 and 
                        solution_colors[neighbor_idx] == color):
                        neighbor_uses_color = True
                        break
                
                # If no neighbor uses this color yet, assigning it would constrain this node
                if not neighbor_uses_color:
                    constraint_count += 1.0
        
        constraining_scores[i] = constraint_count
    
    return constraining_scores


# ============================================================================
# ACO CLASS
# ============================================================================


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

    def __init__(self, graph, iterations=30, alpha=1.0, beta=2.0, rho=0.1, ant_count=100, Q=1.0, verbose=False, patience=0.5, trial_number=None, graph_name=None, graph_index=None, tabu_best=None, strategy_rotation_frequency=10, num_strategies=3, color_preference_weight=10.0, constraint_penalty_weight=0.2, pheromone_init_multiplier=2.0, min_pheromone=0.01, max_pheromone=10.0, warmup_cache_dir=None, warmup_threads=10, use_warmup=True, exploitation_ratio=0.5):
        """
        Initialize ACO Graph Coloring algorithm.
        
        Args:
            graph: NetworkX Graph object
            iterations: Number of iterations to run (default: 30)
            alpha: Pheromone importance - higher = more exploitation (default: 1.0)
            beta: Heuristic importance - higher = more greedy (default: 2.0)
            rho: Evaporation rate - higher = less memory, more exploration (default: 0.1)
            ant_count: Total number of ants per iteration (default: 100)
            Q: Pheromone deposit intensity (default: 1.0)
            verbose: Print progress information (default: False)
            patience: Fraction of iterations without improvement before early stopping (default: 0.5)
            tabu_best: Best known color count from tabu search (for comparison)
            strategy_rotation_frequency: Change node ordering strategy every N iterations (default: 10)
            num_strategies: Number of different node ordering strategies (2-3, default: 3)
            color_preference_weight: Preference for already used colors (default: 10.0)
            constraint_penalty_weight: Weight for constraint penalty in heuristic (default: 0.2)
            pheromone_init_multiplier: Multiplier for initial pheromone seeding (default: 2.0)
            min_pheromone: Minimum pheromone value (prevents stagnation, default: 0.01)
            max_pheromone: Maximum pheromone value (prevents dominance, default: 10.0)
            use_warmup: Enable warm-up phase (default: True, set False to skip warm-up)
            exploitation_ratio: Ratio of pheromone-following ants (0.0-1.0, default: 0.5)
                               Higher = more exploitation, Lower = more exploration
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        
        # Estimate initial number of colors (upper bound)
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
        
        # ANT STRATEGY (Exploitation vs Exploration Balance):
        # ===================================================
        # Ants are split into two groups controlled by exploitation_ratio parameter:
        
        # Group 1: PHEROMONE-FOLLOWING ANTS (exploitation_ratio * ant_count)
        # Purpose: Exploit learned knowledge, intensify search
        # Behavior:
        #   - Start from high-degree nodes (weighted by degree)
        #   - Follow pheromone trails (guided by alpha parameter)
        #   - Probabilistic color selection: pheromone^alpha * heuristic^beta
        #   - Refine and improve known good solutions
        
        # Group 2: ANTI-PHEROMONE EXPLORATION ANTS ((1-exploitation_ratio) * ant_count)
        # Purpose: Explore unexplored regions, diversify search
        # Behavior:
        #   - Random starting nodes for diversity
        #   - INVERTED pheromones (prefer unexplored paths)
        #   - Probabilistic selection with anti-pheromone bias
        #   - Discover alternative solution spaces
        #   - Escape local optima
        
        # Balance is controlled by:
        # - exploitation_ratio: ratio of pheromone vs anti-pheromone ants
        # - alpha: pheromone importance (higher = more exploitation)
        # - rho: evaporation rate (higher = less memory = more exploration)
        # - beta: heuristic importance (higher = more greedy)
        
        self.exploitation_ratio = max(0.0, min(1.0, exploitation_ratio))  # Clamp to [0, 1]
        self.pheromone_ants = int(ant_count * self.exploitation_ratio)
        self.anti_pheromone_ants = ant_count - self.pheromone_ants
        
        # Configurable parameters (no hard-coded values!)
        self.strategy_rotation_frequency = strategy_rotation_frequency
        self.num_strategies = min(max(num_strategies, 2), 3)  # Clamp between 2-3
        self.color_preference_weight = color_preference_weight
        self.constraint_penalty_weight = constraint_penalty_weight
        self.pheromone_init_multiplier = pheromone_init_multiplier
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.warmup_cache_dir = warmup_cache_dir
        self.warmup_threads = warmup_threads
        self.use_warmup = use_warmup
        
        if patience is None:
            self.patience = iterations  # No early stopping
        else:
            self.patience = int(iterations *  patience)

        # Create node index mapping
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        
        # Initialize pheromone matrix: pheromone[node][color]
        # Start with uniform pheromone levels within min/max bounds
        self.pheromone = np.ones((self.N, self.max_colors), dtype=float)
        
        # Initialize node-to-node pheromone matrix for post-warmup phase
        # This represents transition preferences: which node to color next
        # Will be initialized from best warm-up solution
        self.node_to_node_pheromone = None
        self.warmup_complete = False
        
        # Build adjacency list for faster neighbor lookup
        self.adjacency = {node: set(self.graph.neighbors(node)) for node in self.nodes}
        
        # Precompute node degrees for ordering heuristic
        self.node_degrees = {node: len(self.adjacency[node]) for node in self.nodes}
        
        # Build adjacency matrix for vectorized operations (performance optimization)
        self.adj_matrix = nx.to_numpy_array(graph, nodelist=self.nodes, dtype=np.int8)
        
        # Seed pheromones with greedy solution for faster convergence
        self._seed_pheromones_with_greedy()
    
    def _initialize_node_to_node_pheromones(self, solution):
        """
        Initialize node-to-node pheromone matrix from best warm-up solution.
        
        This creates a NxN matrix representing transition preferences:
        - High pheromone on transitions that appeared in good solutions
        - Uniform baseline for unexplored transitions
        
        Args:
            solution: Best solution from warm-up phase (node -> color)
        """
        # Initialize with uniform baseline
        self.node_to_node_pheromone = np.ones((self.N, self.N), dtype=float) * self.min_pheromone
        
        # Reconstruct coloring order from solution using saturation degree
        colored = {}
        uncolored = set(self.nodes)
        
        coloring_order = []
        
        while uncolored:
            # Find next node to color (highest saturation)
            best_node = None
            best_saturation = -1
            best_degree = -1
            
            for node in uncolored:
                neighbor_colors = {solution[n] for n in self.adjacency[node] if n in colored}
                saturation = len(neighbor_colors)
                degree = self.node_degrees[node]
                
                if saturation > best_saturation or (saturation == best_saturation and degree > best_degree):
                    best_node = node
                    best_saturation = saturation
                    best_degree = degree
            
            if best_node is None:
                best_node = next(iter(uncolored))
            
            coloring_order.append(best_node)
            colored[best_node] = solution[best_node]
            uncolored.remove(best_node)
        
        # Deposit pheromones on transitions in coloring order
        num_colors = len(set(solution.values()))
        deposit = self.Q * self.pheromone_init_multiplier / num_colors
        
        for i in range(len(coloring_order) - 1):
            from_node = coloring_order[i]
            to_node = coloring_order[i + 1]
            from_idx = self.node_index[from_node]
            to_idx = self.node_index[to_node]
            
            # Reinforce this transition
            self.node_to_node_pheromone[from_idx][to_idx] += deposit
            
            # Also reinforce transitions from ALL previously colored nodes to next node
            # This captures the \"attractiveness\" of coloring a node at a certain stage
            for j in range(i + 1):
                prev_node = coloring_order[j]
                prev_idx = self.node_index[prev_node]
                self.node_to_node_pheromone[prev_idx][to_idx] += deposit * 0.5
        
        # Apply bounds
        self._apply_pheromone_bounds()
        
        if self.verbose:
            print(f"✅ Node-to-node pheromones initialized from best warm-up solution ({num_colors} colors)")
    
    def _seed_pheromones_with_greedy(self):
        """
        Seed pheromone matrix with a greedy DSatur solution for faster convergence.
        Uses proper DSatur ordering for better initial coloring.
        Returns the greedy solution for initialization.
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
        deposit = self.Q * self.pheromone_init_multiplier / num_colors  # Configurable boost for initial seeding
        
        for node, color in solution.items():
            node_idx = self.node_index[node]
            if color < self.max_colors:  # Only if color is within current range
                self.pheromone[node_idx][color] += deposit
        
        # Store as initial best solution
        self.best_global_solution = solution
        
        return solution
        
        # Apply pheromone bounds after seeding
        self._apply_pheromone_bounds()

    def _get_node_ordering(self, start_node, solution, iteration=0, ant_id=0, forced_strategy=None):
        """
        Get node ordering using strategy-based approach.
        
        STRATEGY SELECTION:
        ------------------
        - Greedy ants: Use forced_strategy parameter (ensures systematic coverage)
        - Other ants: Use iteration-based rotation
        
        AVAILABLE STRATEGIES:
        --------------------
        Strategy 0: Pure DSatur (saturation-first, then degree)
                   Best for dense graphs, prioritizes conflict resolution
                   
        Strategy 1: Degree-first (Welsh-Powell style)
                   Best for sparse graphs, prioritizes high-degree nodes
                   
        Strategy 2: Balanced (saturation with controlled randomization)
                   Good for exploration, adds diversity
        
        Args:
            start_node: Starting node
            solution: Current partial solution
            iteration: Current iteration number
            ant_id: Ant identifier
            forced_strategy: For greedy ants, force specific strategy (0, 1, or 2)
            
        Returns:
            List of nodes in order to be colored
        """
        uncolored = [n for n in self.nodes if n not in solution]
        
        # Pre-calculate saturations for all uncolored nodes
        saturations = []
        for node in uncolored:
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            saturation = len(neighbor_colors)
            degree = self.node_degrees[node]
            saturations.append((node, saturation, degree))
        
        # Determine which strategy to use
        if forced_strategy is not None:
            # Greedy ants: use forced strategy for systematic coverage
            strategy = forced_strategy
        else:
            # Other ants: rotate based on iteration
            strategy = (iteration // self.strategy_rotation_frequency) % self.num_strategies
        
        # Apply selected strategy
        if strategy == 0:
            # Strategy 0: Pure DSatur (saturation first, then degree)
            saturations.sort(key=lambda x: (-x[1], -x[2], ant_id * 0.001))
        elif strategy == 1:
            # Strategy 1: Degree-first (Welsh-Powell style)
            saturations.sort(key=lambda x: (-x[2], -x[1], ant_id * 0.001))
        else:
            # Strategy 2: Balanced with controlled randomization
            saturations.sort(key=lambda x: (-x[1], -x[2], random.random() * 0.1 + ant_id * 0.001))
        
        return [node for node, sat, deg in saturations]
    
    def _ant_construct_solution(self, start_node, iteration=0, ant_id=0, ant_type=AntType.PHEROMONE, forced_strategy=None):
        """
        Construct a solution for one ant using constructive approach.
        
        ANT TYPE BEHAVIORS:
        -------------------
        AntType.GREEDY: Strategy-driven deterministic construction
                 - Uses forced_strategy parameter for node ordering
                 - Reduced pheromone importance (strategy > pheromone)
                 - Deterministic color selection (best score)
                 
        AntType.PHEROMONE: Standard ACO with high pheromone importance
                    - Normal pheromone importance (alpha)
                    - Probabilistic color selection
                    - Exploits learned knowledge
                    
        AntType.ANTI_PHEROMONE: Exploration with inverted pheromones
                         - Inverted pheromone values (unexplored paths preferred)
                         - Probabilistic color selection
                         - Escapes local optima
        
        Args:
            start_node: Starting node for this ant
            iteration: Current iteration number
            ant_id: Ant identifier
            ant_type: Type of ant (use AntType constants)
            forced_strategy: For greedy ants, force specific strategy (Strategy constants)
            
        Returns:
            Dictionary mapping nodes to colors
        """
        solution = {}
        used_colors = set()
        
        # Start with the start_node
        current_node = start_node
        
        # Color nodes iteratively using dynamic ordering
        while len(solution) < self.N:
            # Get next batch of nodes to consider (dynamic ordering based on iteration)
            if current_node is not None and current_node not in solution:
                node = current_node
                current_node = None  # Only use once
            else:
                # Get dynamically ordered uncolored nodes
                # Greedy ants use forced_strategy for systematic coverage
                nodes_order = self._get_node_ordering(start_node, solution, iteration, ant_id, forced_strategy)
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
            
            # Calculate scores ONLY for valid colors with enhanced heuristic (NUMBA OPTIMIZED)
            num_valid = len(valid_colors)
            valid_colors_arr = np.array(valid_colors, dtype=np.int32)
            
            # Build solution color array for fast lookup
            solution_colors = np.full(self.N, -1, dtype=np.int32)
            for solved_node, solved_color in solution.items():
                solution_colors[self.node_index[solved_node]] = solved_color
            
            # Get uncolored neighbors of current node
            uncolored_neighbor_indices = []
            for neighbor in self.adjacency[node]:
                if neighbor not in solution:
                    uncolored_neighbor_indices.append(self.node_index[neighbor])
            uncolored_neighbor_indices = np.array(uncolored_neighbor_indices, dtype=np.int32)
            
            # Calculate constraint scores using Numba JIT
            if len(uncolored_neighbor_indices) > 0:
                constraint_scores = calculate_constraint_scores_numba(
                    valid_colors_arr, solution_colors, self.adj_matrix, 
                    uncolored_neighbor_indices
                )
            else:
                constraint_scores = np.zeros(num_valid, dtype=np.float32)
            
            # Convert used_colors to array for Numba
            used_colors_arr = np.array(list(used_colors), dtype=np.int32)
            
            # Calculate final scores using Numba JIT
            # Adjust pheromone importance based on ant type
            pheromone_values = self.pheromone[node_idx].copy()
            alpha_adjusted = self.alpha
            
            if ant_type == AntType.GREEDY:
                # Greedy ants: REDUCE pheromone importance (strategy > pheromone)
                # Use alpha/5 to make strategy and heuristics dominate
                alpha_adjusted = self.alpha * 0.2
            elif ant_type == AntType.ANTI_PHEROMONE:
                # Anti-pheromone ants: INVERT pheromones (explore unexplored)
                # high pheromone → low score, low pheromone → high score
                pheromone_values = self.max_pheromone - pheromone_values + self.min_pheromone
            # else: pheromone ants use normal alpha (full pheromone importance)
            
            scores = calculate_color_scores_numba(
                pheromone_values, valid_colors_arr, used_colors_arr,
                constraint_scores, alpha_adjusted, self.beta,
                self.color_preference_weight, self.constraint_penalty_weight
            )
            
            # Normalize to probabilities
            score_sum = scores.sum()
            
            # Select color based on ant type
            if ant_type == AntType.GREEDY:
                # Greedy ant: deterministic selection (best score)
                chosen_idx = np.argmax(scores)
                chosen_color = valid_colors[chosen_idx]
            else:
                # Pheromone/anti-pheromone ants: probabilistic selection
                if score_sum > 0:
                    probabilities = scores / score_sum
                else:
                    # Fallback: uniform distribution over valid colors
                    probabilities = np.ones(len(valid_colors)) / len(valid_colors)
                
                chosen_idx = np.random.choice(len(valid_colors), p=probabilities)
                chosen_color = valid_colors[chosen_idx]
            
            solution[node] = int(chosen_color)
            used_colors.add(chosen_color)
        
        return solution

    def _ant_construct_solution_postwarmup(self, start_node, iteration=0, ant_id=0, ant_type=AntType.PHEROMONE):
        """
        Construct a solution using node-to-node pheromone transitions (POST-WARMUP ONLY).
        
        NEW STRATEGY:
        1. Select next node based on node-to-node pheromone transitions
        2. Choose color using least constraining heuristic (leaves most options for neighbors)
        
        ANT TYPE BEHAVIORS:
        AntType.PHEROMONE: Follow node-to-node pheromones (high -> high transitions)
        AntType.ANTI_PHEROMONE: Explore unexplored transitions (low -> high transitions)
        
        Args:
            start_node: Starting node
            iteration: Current iteration number
            ant_id: Ant identifier
            ant_type: Type of ant (use AntType constants)
            
        Returns:
            Dictionary mapping nodes to colors
        """
        solution = {}
        used_colors = set()
        
        # Start with start_node
        colored = [start_node]
        uncolored = [n for n in self.nodes if n != start_node]
        
        # Color all nodes
        for _ in range(self.N):
            if len(colored) == 0:
                # First node
                node = start_node
            elif len(uncolored) == 0:
                break
            else:
                # Select next node using node-to-node pheromones
                # Calculate transition scores from all colored nodes to each uncolored node
                transition_scores = np.zeros(len(uncolored), dtype=np.float64)
                
                for i, target_node in enumerate(uncolored):
                    target_idx = self.node_index[target_node]
                    
                    # Aggregate pheromones from all colored nodes to this target
                    total_pheromone = 0.0
                    for source_node in colored:
                        source_idx = self.node_index[source_node]
                        pheromone_value = self.node_to_node_pheromone[source_idx][target_idx]
                        
                        # Apply ant-type specific logic
                        if ant_type == AntType.ANTI_PHEROMONE:
                            # Invert pheromones for exploration
                            pheromone_value = self.max_pheromone - pheromone_value + self.min_pheromone
                        
                        total_pheromone += pheromone_value
                    
                    # Apply pheromone importance (alpha)
                    transition_scores[i] = total_pheromone ** self.alpha
                
                # Select next node probabilistically based on transition scores
                score_sum = transition_scores.sum()
                if score_sum > 0:
                    probabilities = transition_scores / score_sum
                else:
                    probabilities = np.ones(len(uncolored)) / len(uncolored)
                
                chosen_idx = np.random.choice(len(uncolored), p=probabilities)
                node = uncolored[chosen_idx]
            
            # Now color the selected node using LEAST CONSTRAINING heuristic
            node_idx = self.node_index[node]
            
            # Find valid colors (no conflicts with neighbors)
            neighbor_colors = set()
            for neighbor in self.adjacency[node]:
                if neighbor in solution:
                    neighbor_colors.add(solution[neighbor])
            
            # Candidate colors
            max_used_color = max(used_colors) if used_colors else -1
            candidate_colors = list(range(max_used_color + 2))
            
            if candidate_colors and max(candidate_colors) >= self.max_colors:
                self._expand_pheromone_matrix(max(candidate_colors) + 1)
            
            valid_colors = [c for c in candidate_colors if c not in neighbor_colors]
            
            if not valid_colors:
                new_color = max_used_color + 1
                if new_color >= self.max_colors:
                    self._expand_pheromone_matrix(new_color + 1)
                valid_colors = [new_color]
            
            # Calculate LEAST CONSTRAINING scores (lower = better)
            valid_colors_arr = np.array(valid_colors, dtype=np.int32)
            
            # Build solution color array
            solution_colors = np.full(self.N, -1, dtype=np.int32)
            for solved_node, solved_color in solution.items():
                solution_colors[self.node_index[solved_node]] = solved_color
            
            # Get all uncolored node indices
            uncolored_indices = np.array([self.node_index[n] for n in uncolored if n != node], dtype=np.int32)
            
            # Calculate least constraining scores using Numba
            if len(uncolored_indices) > 0:
                constraining_scores = calculate_least_constraining_scores_numba(
                    valid_colors_arr, solution_colors, self.adj_matrix,
                    node_idx, uncolored_indices
                )
            else:
                constraining_scores = np.zeros(len(valid_colors), dtype=np.float32)
            
            # Convert used_colors to array for Numba
            used_colors_arr = np.array(list(used_colors), dtype=np.int32)
            
            # Combine with color preference heuristic
            color_scores = np.zeros(len(valid_colors), dtype=np.float64)
            for i, color in enumerate(valid_colors):
                # Prefer used colors over new colors
                color_preference = self.color_preference_weight if color in used_colors_arr else 1.0
                
                # Lower constraining score = better (invert for scoring)
                # Add small constant to avoid division by zero
                least_constraining_bonus = 1.0 / (1.0 + constraining_scores[i] * 0.5)
                
                color_scores[i] = (color_preference * least_constraining_bonus) ** self.beta
            
            # Select color probabilistically
            score_sum = color_scores.sum()
            if score_sum > 0:
                probabilities = color_scores / score_sum
            else:
                probabilities = np.ones(len(valid_colors)) / len(valid_colors)
            
            chosen_idx = np.random.choice(len(valid_colors), p=probabilities)
            chosen_color = valid_colors[chosen_idx]
            
            # Assign color
            solution[node] = int(chosen_color)
            used_colors.add(chosen_color)
            colored.append(node)
            if node in uncolored:
                uncolored.remove(node)
        
        return solution

    def _evaluate_solution(self, solution):
        """
        Evaluate a coloring solution.
        
        Returns:
            Number of colors used
        """
        return len(set(solution.values()))

    def _ant_worker_thread(self, start_node, results_queue, ant_id, iteration, ant_type, forced_strategy=None):
        """
        Worker function for each ant (runs in separate thread).
        
        Args:
            start_node: Starting node
            results_queue: Queue to store results
            ant_id: Ant identifier
            iteration: Current iteration number
            ant_type: Type of ant (use AntType constants)
            forced_strategy: For greedy ants, force a specific strategy (Strategy constants)
        """
        # Use different construction method based on phase
        if self.warmup_complete and ant_type in [AntType.PHEROMONE, AntType.ANTI_PHEROMONE]:
            # Post-warmup: use node-to-node pheromone construction
            solution = self._ant_construct_solution_postwarmup(start_node, iteration, ant_id, ant_type)
        else:
            # Warm-up or greedy ant: use original construction
            solution = self._ant_construct_solution(start_node, iteration, ant_id, ant_type, forced_strategy)
        
        num_colors = self._evaluate_solution(solution)
        
        results_queue.put({
            'solution': solution,
            'num_colors': num_colors,
            'ant_type': ant_type  # Track which ant type found this solution
        })

    def _expand_pheromone_matrix(self, new_max_colors):
        """
        Expand pheromone matrix to accommodate more colors.
        New columns are initialized with minimum pheromone value.
        
        Args:
            new_max_colors: New maximum number of colors
        """
        if new_max_colors <= self.max_colors:
            return
        
        # Create new columns initialized with minimum pheromone value
        additional_colors = new_max_colors - self.max_colors
        new_columns = np.full((self.N, additional_colors), self.min_pheromone, dtype=float)
        
        # Append new columns to existing pheromone matrix
        self.pheromone = np.hstack([self.pheromone, new_columns])
        self.max_colors = new_max_colors
    
    def _apply_pheromone_bounds(self):
        """
        Apply min/max bounds to pheromone matrix to prevent stagnation and dominance.
        
        This prevents:
        - Stagnation: pheromones going to zero, making paths effectively invisible
        - Dominance: pheromones becoming too large, eliminating exploration
        """
        np.clip(self.pheromone, self.min_pheromone, self.max_pheromone, out=self.pheromone)
        
        # Also apply bounds to node-to-node pheromone if initialized
        if self.node_to_node_pheromone is not None:
            np.clip(self.node_to_node_pheromone, self.min_pheromone, self.max_pheromone, 
                    out=self.node_to_node_pheromone)
    
    def _evaporate_pheromones(self):
        """
        Evaporate pheromones globally by factor (1 - rho).
        Applies bounds after evaporation to maintain healthy pheromone levels.
        """
        self.pheromone *= (1.0 - self.rho)
        
        # Also evaporate node-to-node pheromones in post-warmup phase
        if self.node_to_node_pheromone is not None:
            self.node_to_node_pheromone *= (1.0 - self.rho)
        
        self._apply_pheromone_bounds()

    def _reinforce_pheromones(self, solution, num_colors):
        """
        Deposit pheromones on the given solution.
        
        In post-warmup phase, reinforces node-to-node transitions based on coloring order.
        Deposit amount inversely proportional to number of colors used.
        Applies bounds after reinforcement to prevent dominance.
        
        Args:
            solution: Solution dict (node -> color)
            num_colors: Number of colors in solution
        """
        deposit = self.Q / num_colors
        
        # Node×color pheromones (legacy - kept for greedy ants during warm-up)
        for node, color in solution.items():
            node_idx = self.node_index[node]
            self.pheromone[node_idx][color] += deposit
        
        # Node-to-node transition pheromones (post-warmup phase)
        if self.node_to_node_pheromone is not None and self.warmup_complete:
            # Reconstruct coloring order and reinforce transitions
            # Use saturation degree ordering to approximate coloring sequence
            colored_nodes = set()
            uncolored_nodes = set(solution.keys())
            coloring_sequence = []
            
            while uncolored_nodes:
                # Find next node to color (highest saturation)
                best_node = None
                best_saturation = -1
                best_degree = -1
                
                for node in uncolored_nodes:
                    neighbor_colors = {solution[n] for n in self.adjacency[node] if n in colored_nodes}
                    saturation = len(neighbor_colors)
                    degree = self.node_degrees[node]
                    
                    if saturation > best_saturation or (saturation == best_saturation and degree > best_degree):
                        best_node = node
                        best_saturation = saturation
                        best_degree = degree
                
                if best_node is None:
                    best_node = next(iter(uncolored_nodes))
                
                # Reinforce transitions from all colored nodes to this node
                for from_node in colored_nodes:
                    from_idx = self.node_index[from_node]
                    to_idx = self.node_index[best_node]
                    self.node_to_node_pheromone[from_idx][to_idx] += deposit
                
                colored_nodes.add(best_node)
                uncolored_nodes.remove(best_node)
                coloring_sequence.append(best_node)
        
        # Apply bounds after reinforcement
        self._apply_pheromone_bounds()


    def run(self, use_warmup_cache=True):
        """
        Run the ACO algorithm.
        
        Args:
            use_warmup_cache: Use cached warm-up results if available (default: True)
        
        Returns:
            Dictionary with:
                - solution: Best solution found (node -> color mapping)
                - color_count: Number of colors in best solution
                - iterations: Number of iterations executed
                - warmup_result: Result from warm-up phase (None if warmup disabled)
        """
        # ========================================================================
        # WARM-UP PHASE (OPTIONAL, SEPARATE, NOT COUNTED IN ITERATIONS)
        # ========================================================================
        # Run comprehensive greedy exploration using separate module
        # This is cached and reused across runs for the same graph
        
        warmup_result = None
        
        if self.use_warmup:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"PHASE 1: WARM-UP (Greedy Exploration)")
                print(f"{'='*80}")
            
            warmup_result = run_warmup(
                graph=self.graph,
                num_strategies=self.num_strategies,
                verbose=self.verbose,
                cache_dir=self.warmup_cache_dir,
                num_threads=self.warmup_threads,
                use_cache=use_warmup_cache
            )
            
            # Initialize from warm-up result
            self.best_global_solution = warmup_result['solution']
            best_global_colors = warmup_result['color_count']
            
            # Initialize node-to-node pheromones from warm-up solution
            self._initialize_node_to_node_pheromones(warmup_result['solution'])
            self.warmup_complete = True
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"PHASE 2: ACO REFINEMENT ({self.iterations} iterations)")
                print(f"{'='*80}")
                print(f"Starting with {best_global_colors} colors from warm-up")
                print(f"Strategy: {int(self.exploitation_ratio*100)}% pheromone + {int((1-self.exploitation_ratio)*100)}% anti-pheromone ants\n")
        else:
            # Skip warm-up, start with greedy seed solution
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"ACO OPTIMIZATION ({self.iterations} iterations)")
                print(f"{'='*80}")
                print(f"Warm-up phase disabled - starting with greedy seed")
                print(f"Strategy: {int(self.exploitation_ratio*100)}% pheromone + {int((1-self.exploitation_ratio)*100)}% anti-pheromone ants\n")
            
            # Use the greedy seed solution already initialized in __init__
            best_global_colors = len(set(self.best_global_solution.values()))
            
            # Initialize node-to-node pheromones from seed solution
            self._initialize_node_to_node_pheromones(self.best_global_solution)
            self.warmup_complete = True
        
        iterations_without_improvement = 0
        
        for iteration in range(1, self.iterations + 1):
            # Create ants and run
            results_queue = Queue()
            threads = []
            iteration_solutions = []
            
            # ====================================================================
            # ANT DEPLOYMENT: POST-WARMUP ONLY (2-GROUP PHEROMONE STRATEGY)
            # ====================================================================
            # Warm-up is done separately, all iterations here use pheromone guidance
            
            start_nodes = []
            ant_types = []
            
            # Sort nodes by degree for reference
            sorted_nodes = sorted(self.nodes, key=lambda n: self.node_degrees[n], reverse=True)
            
            # Calculate ant group sizes based on exploitation_ratio
            # exploitation_ratio controls the balance:
            # - High ratio (e.g., 0.8) = more exploitation (80% follow pheromones)
            # - Low ratio (e.g., 0.2) = more exploration (80% explore with anti-pheromones)
            # - Default 0.5 = balanced (50/50 split)
            pheromone_group_size = self.pheromone_ants
            anti_pheromone_group_size = self.anti_pheromone_ants
            
            # ----------------------------------------------------------------
            # GROUP 1: PHEROMONE-FOLLOWING ANTS (exploitation_ratio of ants)
            # ----------------------------------------------------------------
            # Exploit learned knowledge, refine best solutions
            
            degrees = [self.node_degrees[n] for n in sorted_nodes]
            total_degree = sum(degrees)
            weights = [d / total_degree for d in degrees] if total_degree > 0 else None
            pheromone_start_nodes = random.choices(sorted_nodes, weights=weights, k=pheromone_group_size)
            
            start_nodes.extend(pheromone_start_nodes)
            ant_types.extend([AntType.PHEROMONE] * len(pheromone_start_nodes))
            
            # ----------------------------------------------------------------
            # GROUP 2: ANTI-PHEROMONE EXPLORATION ANTS ((1-exploitation_ratio) of ants)
            # ----------------------------------------------------------------
            # Explore unexplored regions, escape local optima
            
            anti_start_nodes = random.choices(self.nodes, k=anti_pheromone_group_size)
            
            start_nodes.extend(anti_start_nodes)
            ant_types.extend([AntType.ANTI_PHEROMONE] * len(anti_start_nodes))
            
            # ========================================================================
            # PARALLEL ANT EXECUTION
            # ========================================================================
            # Run all ants in parallel threads
            
            for ant_id, (start_node, ant_type) in enumerate(zip(start_nodes, ant_types)):
                thread = threading.Thread(
                    target=self._ant_worker_thread,
                    args=(start_node, results_queue, ant_id, iteration, ant_type, None)
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
            
            # ====================================================================
            # PHEROMONE UPDATE
            # ====================================================================
            # Evaporate and selectively reinforce pheromones
            
            self._evaporate_pheromones()
            
            # Reinforce pheromones on iteration-best solution
            # (all ant types contribute to pheromone learning)
            self._reinforce_pheromones(iter_best['solution'], iter_best['num_colors'])
            
            # ====================================================================
            # PROGRESS REPORTING
            # ====================================================================
            
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
            if self.verbose and iteration % 10 == 0:
                print(f"💬 {prefix}Iteration {iteration}/{self.iterations}: "
                        f"iter_best={iter_best['num_colors']}, "
                        f"global_best={best_global_colors}{tabu_info}")
            
            # Check early stopping condition
            if self.patience is not None and iterations_without_improvement >= self.patience:
                if self.verbose:
                    print()  # New line after progress updates
                    print(f"❌ {prefix} Early stopping: No improvement for {self.patience} iterations")
                return {
                    'solution': self.best_global_solution,
                    'color_count': best_global_colors,
                    'iterations': iteration,
                    'warmup_result': warmup_result
                }
        
        # Print newline after progress updates
        if self.verbose:
            print()
        
        return {
            'solution': self.best_global_solution,
            'color_count': best_global_colors,
            'iterations': iteration,
            'warmup_result': warmup_result
        }

