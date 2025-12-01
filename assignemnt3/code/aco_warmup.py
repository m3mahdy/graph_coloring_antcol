"""
ACO Warm-up Phase Module

Comprehensive greedy exploration that systematically covers all possible
greedy algorithm variations before starting pheromone-based ACO.

Features:
- Caches warm-up results to disk for reuse across runs
- Tests all node orderings and strategies
- Returns best solution found to initialize pheromones
- Independent of main ACO iterations

Coverage:
1. Standard Greedy: Sequential node ordering (0→N-1) with each strategy
2. S-Start: Each node as starting point
3. Strategy Combinations: Each node with each ordering strategy
"""

import networkx as nx
import numpy as np
import json
import hashlib
from pathlib import Path
from copy import deepcopy
import threading
from queue import Queue


class ACOWarmup:
    """
    Warm-up phase for ACO: exhaustive greedy exploration.
    
    This class handles the systematic exploration of all greedy algorithm
    variations before the main ACO phase begins. Results are cached to disk
    to avoid redundant computation.
    """
    
    def __init__(self, graph, num_strategies=3, verbose=False, cache_dir=None):
        """
        Initialize ACO warm-up phase.
        
        Args:
            graph: NetworkX graph object
            num_strategies: Number of ordering strategies (2-4, default: 3)
            verbose: Print progress information (default: False)
            cache_dir: Directory for caching results (default: ./warmup_cache)
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)
        self.num_strategies = min(max(num_strategies, 2), 4)
        self.verbose = verbose
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / 'warmup_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create node index mapping
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        
        # Build adjacency list and degrees
        self.adjacency = {node: set(graph.neighbors(node)) for node in self.nodes}
        self.node_degrees = {node: len(self.adjacency[node]) for node in self.nodes}
    
    def _get_graph_hash(self):
        """
        Generate unique hash for the graph structure.
        
        Uses graph structure (edges, nodes) to create a deterministic hash
        that identifies this specific graph.
        
        Returns:
            Hash string
        """
        # Create deterministic representation
        edges = sorted([(min(u, v), max(u, v)) for u, v in self.graph.edges()])
        graph_str = f"{self.N}_{len(edges)}_{edges[:100]}"  # Use first 100 edges
        
        return hashlib.md5(graph_str.encode()).hexdigest()
    
    def _get_cache_path(self):
        """Get path to cache file for this graph."""
        graph_hash = self._get_graph_hash()
        return self.cache_dir / f"warmup_{graph_hash}_strategies{self.num_strategies}.json"
    
    def _load_cached_result(self):
        """
        Load cached warm-up result if available.
        
        Returns:
            Cached result dict or None if not found
        """
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            if self.verbose:
                print(f"✅ Loaded cached warm-up result from {cache_path.name}")
                print(f"   Best solution: {cached['best_colors']} colors")
                print(f"   Scenarios tested: {cached['scenarios_tested']}")
            
            return cached
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load cache: {e}")
            return None
    
    def _save_cached_result(self, result):
        """
        Save warm-up result to cache.
        
        Args:
            result: Result dictionary to cache
        """
        cache_path = self._get_cache_path()
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            if self.verbose:
                print(f"💾 Cached warm-up result to {cache_path.name}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to save cache: {e}")
    
    def _get_node_ordering(self, start_node, solution, strategy_idx):
        """
        Get node ordering using specific strategy.
        
        Args:
            start_node: Starting node
            solution: Current partial solution
            strategy_idx: Strategy index (0, 1, 2, or 3)
            
        Returns:
            List of uncolored nodes in priority order
        """
        uncolored = [n for n in self.nodes if n not in solution]
        
        # Pre-calculate saturations for all uncolored nodes
        saturations = []
        for node in uncolored:
            neighbor_colors = {solution[neighbor] for neighbor in self.adjacency[node] 
                             if neighbor in solution}
            saturation = len(neighbor_colors)
            degree = self.node_degrees[node]
            saturations.append((node, saturation, degree))
        
        # Apply selected strategy
        if strategy_idx == 0:
            # Strategy 0: Pure DSatur (saturation first, then degree)
            saturations.sort(key=lambda x: (-x[1], -x[2]))
        elif strategy_idx == 1:
            # Strategy 1: Degree-first (Welsh-Powell style)
            saturations.sort(key=lambda x: (-x[2], -x[1]))
        elif strategy_idx == 2:
            # Strategy 2: Balanced with light randomization
            saturations.sort(key=lambda x: (-x[1], -x[2], np.random.random() * 0.1))
        else:
            # Strategy 3: Saturation with degree penalty
            saturations.sort(key=lambda x: (-x[1] * 2 - x[2]))
        
        return [node for node, sat, deg in saturations]
    
    def _construct_greedy_solution(self, start_node, strategy_idx=None, scenario_id=0):
        """
        Construct a greedy solution.
        
        Args:
            start_node: Starting node for construction
            strategy_idx: Fixed strategy index, or None for dynamic
            scenario_id: Scenario identifier for tie-breaking
            
        Returns:
            Dictionary mapping nodes to colors
        """
        solution = {}
        used_colors = set()
        current_node = start_node
        
        # Color nodes iteratively
        while len(solution) < self.N:
            # Select node to color
            if current_node is not None and current_node not in solution:
                node = current_node
                current_node = None
            else:
                # Get dynamically ordered uncolored nodes
                nodes_order = self._get_node_ordering(start_node, solution, 
                                                      strategy_idx if strategy_idx is not None else 0)
                if not nodes_order:
                    break
                node = nodes_order[0]
            
            # Find valid colors (no conflicts with neighbors)
            neighbor_colors = {solution[neighbor] for neighbor in self.adjacency[node]
                             if neighbor in solution}
            
            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            solution[node] = color
            used_colors.add(color)
        
        return solution
    
    def _worker_thread(self, scenarios, results_queue):
        """
        Worker thread to process multiple scenarios.
        
        Args:
            scenarios: List of (start_node, strategy_idx, scenario_id) tuples
            results_queue: Queue to store results
        """
        for start_node, strategy_idx, scenario_id in scenarios:
            solution = self._construct_greedy_solution(start_node, strategy_idx, scenario_id)
            num_colors = len(set(solution.values()))
            
            results_queue.put({
                'solution': solution,
                'num_colors': num_colors,
                'scenario_id': scenario_id
            })
    
    def run(self, num_threads=10, use_cache=True):
        """
        Run comprehensive warm-up phase.
        
        Args:
            num_threads: Number of parallel threads (default: 10)
            use_cache: Load/save cached results (default: True)
            
        Returns:
            Dictionary with:
                - solution: Best solution found (node -> color)
                - color_count: Number of colors in best solution
                - scenarios_tested: Total scenarios explored
                - from_cache: Whether result was loaded from cache
        """
        # Try to load from cache
        if use_cache:
            cached = self._load_cached_result()
            if cached is not None:
                return {
                    'solution': {int(k): v for k, v in cached['solution'].items()},
                    'color_count': cached['best_colors'],
                    'scenarios_tested': cached['scenarios_tested'],
                    'from_cache': True
                }
        
        if self.verbose:
            print(f"\n🔥 Starting Warm-up Phase")
            print(f"   Graph: {self.N} nodes, {self.graph.number_of_edges()} edges")
            print(f"   Strategies: {self.num_strategies}")
        
        # Calculate scenarios to cover
        standard_greedy_scenarios = self.num_strategies
        sstart_scenarios = self.N
        strategy_scenarios = self.N * self.num_strategies
        total_scenarios = standard_greedy_scenarios + sstart_scenarios + strategy_scenarios
        
        if self.verbose:
            print(f"   Total scenarios: {total_scenarios}")
            print(f"     - Standard greedy: {standard_greedy_scenarios}")
            print(f"     - S-start: {sstart_scenarios}")
            print(f"     - Strategy combinations: {strategy_scenarios}")
        
        # Build scenario list
        scenarios = []
        scenario_id = 0
        
        # 1. Standard greedy scenarios (sequential 0→N-1 with each strategy)
        for strategy_idx in range(self.num_strategies):
            scenarios.append((self.nodes[0], strategy_idx, scenario_id))
            scenario_id += 1
        
        # 2. S-start scenarios (each node as starting point, dynamic ordering)
        for node_idx in range(self.N):
            scenarios.append((self.nodes[node_idx], None, scenario_id))
            scenario_id += 1
        
        # 3. Strategy combinations (each node with each strategy)
        for node_idx in range(self.N):
            for strategy_idx in range(self.num_strategies):
                scenarios.append((self.nodes[node_idx], strategy_idx, scenario_id))
                scenario_id += 1
        
        # Distribute scenarios across threads
        scenarios_per_thread = (len(scenarios) + num_threads - 1) // num_threads
        threads = []
        results_queue = Queue()
        
        if self.verbose:
            print(f"   Using {num_threads} threads...")
        
        for i in range(num_threads):
            start_idx = i * scenarios_per_thread
            end_idx = min(start_idx + scenarios_per_thread, len(scenarios))
            if start_idx >= len(scenarios):
                break
            
            thread_scenarios = scenarios[start_idx:end_idx]
            thread = threading.Thread(target=self._worker_thread, 
                                     args=(thread_scenarios, results_queue))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Collect results
        best_solution = None
        best_colors = float('inf')
        
        results_collected = 0
        while not results_queue.empty():
            result = results_queue.get()
            results_collected += 1
            
            if result['num_colors'] < best_colors:
                best_solution = result['solution']
                best_colors = result['num_colors']
            
            if self.verbose and results_collected % 100 == 0:
                print(f"   Processed {results_collected}/{total_scenarios} scenarios, "
                      f"best: {best_colors} colors")
        
        if self.verbose:
            print(f"\n✅ Warm-up Complete!")
            print(f"   Best solution: {best_colors} colors")
            print(f"   Scenarios tested: {total_scenarios}")
        
        # Cache result
        result_dict = {
            'solution': {str(k): v for k, v in best_solution.items()},
            'best_colors': best_colors,
            'scenarios_tested': total_scenarios,
            'graph_nodes': self.N,
            'graph_edges': self.graph.number_of_edges()
        }
        
        if use_cache:
            self._save_cached_result(result_dict)
        
        return {
            'solution': best_solution,
            'color_count': best_colors,
            'scenarios_tested': total_scenarios,
            'from_cache': False
        }


def run_warmup(graph, num_strategies=3, verbose=False, cache_dir=None, num_threads=10, use_cache=True):
    """
    Convenience function to run warm-up phase.
    
    Args:
        graph: NetworkX graph object
        num_strategies: Number of ordering strategies (default: 3)
        verbose: Print progress (default: False)
        cache_dir: Cache directory path (default: ./warmup_cache)
        num_threads: Number of parallel threads (default: 10)
        use_cache: Use cached results if available (default: True)
        
    Returns:
        Dictionary with warm-up results
    """
    warmup = ACOWarmup(graph, num_strategies=num_strategies, verbose=verbose, cache_dir=cache_dir)
    return warmup.run(num_threads=num_threads, use_cache=use_cache)
