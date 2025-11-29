"""
ACO Visualization module - handles all visualization-related functionality.
Separated from core ACO algorithm for modularity.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from pathlib import Path


class ACOVisualizer:
    """Handles visualization of ACO graph coloring process."""
    
    def __init__(self, aco_instance):
        """
        Initialize visualizer with ACO instance.
        
        Args:
            aco_instance: ACOGraphColoring instance to visualize
        """
        self.aco = aco_instance
        self.graph = aco_instance.graph
        self.N = aco_instance.N
        self.max_colors = aco_instance.max_colors
        self.pheromone = aco_instance.pheromone
    
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
        Plot a colored graph showing ant's solution.
        
        Args:
            solution: Dictionary mapping node -> color
            ant_id: Ant identifier
            iteration: Current iteration number
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create color map
        num_colors = len(set(solution.values()))
        color_map = cm.get_cmap('tab20', max(num_colors, 3))
        
        # Map node colors
        node_colors = [color_map(solution[node] % 20) for node in self.graph.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(self.graph, seed=42, k=0.5, iterations=50)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                               node_size=300, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1.5, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'Ant {ant_id} Solution - Iteration {iteration}\n{num_colors} colors used', 
                    fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_partial_solution(self, solution, colored_nodes, ant_id, iteration, step, save_path=None):
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
        
        ax.set_title(f'Ant {ant_id} - Iteration {iteration} - Step {step+1}/{len(self.aco.nodes)}\n'
                     f'{len(colored_nodes)} nodes colored, {num_colors} colors used', 
                     fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_best_solution(self, best_solution, iteration, save_path=None):
        """
        Plot the current best global solution.
        
        Args:
            best_solution: Best solution dictionary
            iteration: Current iteration number (for title)
            save_path: Optional path to save the figure
        """
        if best_solution is None:
            print("No solution available to plot.")
            return
        
        self.plot_ant_solution(best_solution, "Best", iteration, save_path)
