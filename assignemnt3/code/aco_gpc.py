"""
ACO Graph Coloring (Parallel Ants)
- Each ant starts from a different node and runs in its own Thread
- Heuristic based on number of conflicts (fewer conflicts => higher heuristic value)
- Pheromone is updated once per iteration using the iteration-best solution
- Pheromone deposit uses: deposit_amount = Q / color_count
- Automatically expands colors if stagnated with conflicts

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
    Ant Colony Optimization for Graph Coloring with parallel ants.

    Notes:
      - Nodes in the supplied NetworkX graph can be any hashable labels.
      - Internally we map node labels to integer indices to store pheromone.
      - Each ant builds a complete coloring solution probabilistically using
        pheromone and heuristic (heuristic here is 1/(1+conflicts)).
      - After all ants finish an iteration, we pick the iteration-best solution
        (min number of colors, tie-breaker: min conflicts) and deposit pheromone.
      - Automatically expands available colors if stagnated with conflicts.
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
            verbose: Print progress information (default: True)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        
        # Set default num_colors with better heuristic for small graphs
        if self.N < 20:
            num_colors = max(3, self.N // 2)  # For small graphs, use minimum 3 colors
        else:
            num_colors = max(10, self.N // 2)  # For larger graphs, use minimum 10 colors
        
        self.initial_num_colors = num_colors
        self.num_colors = num_colors

        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ant_count = ant_count
        self.Q = Q
        self.verbose = verbose

        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.pheromone = np.ones((self.N, self.num_colors), dtype=float)
    
    def _expand_colors(self, additional_colors=1):
        """
        Expand the number of available colors and pheromone matrix.
        
        Args:
            additional_colors: Number of colors to add
        """
        self.num_colors += additional_colors
        new_columns = np.ones((self.N, additional_colors), dtype=float)
        self.pheromone = np.hstack([self.pheromone, new_columns])

    def _heuristic_conflicts(self, node_label, color, partial_solution):
        """
        Heuristic that prefers colors that create fewer immediate conflicts.
        Returns 1/(1 + conflicts) to keep values positive and bounded.
        """
        conflicts = 0
        for neigh in self.graph.neighbors(node_label):
            if neigh in partial_solution and partial_solution[neigh] == color:
                conflicts += 1
        return 1.0 / (1.0 + conflicts)

    def _evaluate_solution(self, solution):
        """
        Returns a tuple (color_count, conflict_count).
        - color_count: number of distinct colors used
        - conflict_count: number of edges whose endpoints share the same color
        """
        color_count = len(set(solution.values()))
        conflicts = 0
        for u, v in self.graph.edges():
            if solution[u] == solution[v]:
                conflicts += 1
        return color_count, conflicts

    def _ant_worker(self, start_node_label, return_queue):
        """
        A single ant constructs a full coloring solution.
        Probabilistic selection at each node uses: (tau^alpha) * (eta^beta)
        """
        solution = {}

        nodes_order = self.nodes.copy()
        if start_node_label in nodes_order:
            nodes_order.remove(start_node_label)
            nodes_order.insert(0, start_node_label)
        else:
            random.shuffle(nodes_order)

        for node in nodes_order:
            scores = np.zeros(self.num_colors, dtype=float)
            idx = self.node_index[node]

            for color in range(self.num_colors):
                tau = self.pheromone[idx][color] ** self.alpha
                eta = self._heuristic_conflicts(node, color, solution) ** self.beta
                scores[color] = tau * eta

            if scores.sum() <= 0:
                probs = np.ones(self.num_colors) / self.num_colors
            else:
                probs = scores / scores.sum()

            chosen_color = np.random.choice(range(self.num_colors), p=probs)
            solution[node] = int(chosen_color)

        color_count, conflict_count = self._evaluate_solution(solution)
        return_queue.put({
            "solution": solution,
            "color_count": color_count,
            "conflict_count": conflict_count
        })

    def _update_pheromones(self, best_solution, best_color_count):
        """
        Evaporate pheromone globally and deposit pheromone along the best solution.
        """
        self.pheromone *= (1.0 - self.rho)

        deposit_amount = float(self.Q) / float(max(1, best_color_count))

        for node_label, color in best_solution.items():
            idx = self.node_index[node_label]
            self.pheromone[idx][color] += deposit_amount

    def run(self):
        """
        Run the ACO optimization using the parameters set during initialization.
        
        Returns:
            Dictionary with best_solution, color_count, conflict_count, and iterations
        """
        best_global_solution = None
        best_global_metrics = (float('inf'), float('inf'))
        stagnation_count = 0
        max_stagnation = 10

        for it in range(1, self.iterations + 1):
            q = Queue()
            threads = []

            if self.ant_count <= self.N:
                start_nodes = random.sample(self.nodes, self.ant_count)
            else:
                start_nodes = [random.choice(self.nodes) for _ in range(self.ant_count)]

            for start in start_nodes:
                t = threading.Thread(target=self._ant_worker, args=(start, q))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            iteration_solutions = []
            while not q.empty():
                item = q.get()
                iteration_solutions.append(item)

            iteration_solutions.sort(key=lambda x: (x["color_count"], x["conflict_count"]))
            iter_best = iteration_solutions[0]

            if (iter_best["color_count"], iter_best["conflict_count"]) < best_global_metrics:
                best_global_metrics = (iter_best["color_count"], iter_best["conflict_count"]) 
                best_global_solution = deepcopy(iter_best["solution"])
                stagnation_count = 0
            else:
                stagnation_count += 1

            self._update_pheromones(iter_best["solution"], iter_best["color_count"]) 

            if self.verbose:
                print(f"Iter {it}: iter_best colors={iter_best['color_count']}, conflicts={iter_best['conflict_count']} | global_best colors={best_global_metrics[0]}, conflicts={best_global_metrics[1]}")

            if best_global_metrics[1] == 0:
                if self.verbose:
                    print("Perfect coloring (0 conflicts) found — stopping early.")
                break
            
            if stagnation_count >= max_stagnation and best_global_metrics[1] > 0:
                if self.verbose:
                    print(f"Stagnated with {best_global_metrics[1]} conflicts. Expanding colors from {self.num_colors} to {self.num_colors + 1}")
                self._expand_colors(1)
                stagnation_count = 0

        return {
            "best_solution": best_global_solution,
            "color_count": best_global_metrics[0],
            "conflict_count": best_global_metrics[1],
            "iterations": it
        }

