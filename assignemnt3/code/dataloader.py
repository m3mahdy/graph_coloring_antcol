"""
Data Loader for ACO Graph Coloring
Loads graph datasets from tuning and testing directories for ACO experiments.
"""

import json
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict


class GraphDataLoader:
    """
    Loads graph datasets for tuning and testing ACO algorithms.
    
    Supports loading from:
    - tiny_dataset: Small graphs for quick testing
    - main_dataset: All graphs including large ones
    """
    
    def __init__(self, data_root: str, dataset_name: str):
        """
        Initialize the data loader.
        
        Args:
            data_root: Absolute path to the data directory
            dataset_name: Name of the dataset ('tiny_dataset' or 'main_dataset')
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    def _load_graph_from_file(self, filepath: Path) -> nx.Graph:
        """
        Load a graph from a file using NetworkX edge list reader.
        
        Args:
            filepath: Path to the graph file
            
        Returns:
            NetworkX Graph object
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        graph = nx.read_edgelist(str(filepath))
        return graph
    
    def _get_dataset_files(self, data_split: str) -> list:
        """
        Get list of all graph files in a dataset directory.
        
        Args:
            data_split: 'tuning_dataset' or 'testing_dataset'
            
        Returns:
            Sorted list of Path objects for graph files
        """
        dataset_path = self.data_root / self.dataset_name / data_split
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        graph_files = [f for f in dataset_path.iterdir() if f.is_file()]
        return sorted(graph_files)
    
    def _print_graph_info(self, filename: str, graph: nx.Graph):
        """
        Print information about a single graph.
        
        Args:
            filename: Name of the graph file
            graph: NetworkX Graph object
        """
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        is_connected = nx.is_connected(graph)
        
        print(f"  {filename}:")
        print(f"    Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"    Density: {density:.4f}, Connected: {is_connected}")
    
    def load_tuning_dataset(self) -> list[Tuple[str, nx.Graph]]:
        """
        Load all graphs from the tuning dataset and return as a list.
        Prints summary after loading each graph.
        
        Returns:
            List of tuples (filename, NetworkX Graph)
        """
        print(f"\n{'='*70}")
        print(f"Loading Tuning Dataset: {self.dataset_name}")
        print(f"{'='*70}")
        
        files = self._get_dataset_files("tuning_dataset")
        graphs = []
        
        for filepath in files:
            try:
                graph = self._load_graph_from_file(filepath)
                self._print_graph_info(filepath.name, graph)
                graphs.append((filepath.name, graph))
            except Exception as e:
                print(f"  {filepath.name}: Error - {e}")
        
        print(f"{'='*70}\n")
        return graphs
    
    def load_testing_dataset(self) -> list[Tuple[str, nx.Graph]]:
        """
        Load all graphs from the testing dataset and return as a list.
        Prints summary after loading each graph.
        
        Returns:
            List of tuples (filename, NetworkX Graph)
        """
        print(f"\n{'='*70}")
        print(f"Loading Testing Dataset: {self.dataset_name}")
        print(f"{'='*70}")
        
        files = self._get_dataset_files("testing_dataset")
        graphs = []
        
        for filepath in files:
            try:
                graph = self._load_graph_from_file(filepath)
                self._print_graph_info(filepath.name, graph)
                graphs.append((filepath.name, graph))
            except Exception as e:
                print(f"  {filepath.name}: Error - {e}")
        
        print(f"{'='*70}\n")
        return graphs
    
    def load_best_known_results(self) -> Dict[str, int]:
        """
        Load best known results from testing_best_known.json in dataset root.
        This file contains benchmark results for each graph.
        
        Returns:
            Dictionary mapping graph names to best known color counts
            
        Raises:
            FileNotFoundError: If testing_best_known.json file does not exist
            ValueError: If JSON format is invalid
        """
        best_known_path = self.data_root / self.dataset_name / "testing_best_known.json"
        
        if not best_known_path.exists():
            raise FileNotFoundError(
                f"Best known results file not found: {best_known_path}\n"
                f"Expected location: data/{self.dataset_name}/testing_best_known.json"
            )
        
        try:
            with open(best_known_path, 'r') as f:
                best_known_list = json.load(f)
            
            if not isinstance(best_known_list, list):
                raise ValueError("testing_best_known.json must contain a list of graph entries")
            
            best_known_dict = {}
            for entry in best_known_list:
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid entry in testing_best_known.json: {entry}")
                if 'graph_name' not in entry or 'best_known_colors' not in entry:
                    raise ValueError(f"Missing required fields in entry: {entry}")
                
                best_known_dict[entry['graph_name']] = entry['best_known_colors']
            
            print(f"\n{'='*70}")
            print(f"Loaded Best Known Results: {len(best_known_dict)} graphs")
            print(f"{'='*70}")
            for graph_name, colors in sorted(best_known_dict.items()):
                print(f"  {graph_name}: {colors} colors")
            print(f"{'='*70}\n")
            
            return best_known_dict
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in testing_best_known.json: {e}")


