import networkx as nx
from pathlib import Path
from aco_gpc import ACOGraphColoring
from datetime import datetime

# Create a test graph (simple cycle with diagonal)
graph = nx.Graph()
graph.add_edges_from([
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),  # 5-cycle
    (1, 3), (2, 4)  # Some diagonals to make it interesting
])

print(f"Test graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print()

# Create directory with timestamp for this test run (in same folder as this script)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = Path(__file__).parent
viz_base = script_dir / 'aco_visualizations'
viz_dir = viz_base / f'test_{timestamp}'
viz_dir.mkdir(parents=True, exist_ok=True)

print(f"Visualizations will be saved to: {viz_dir}")
print()

# Create ACO with visualization enabled
aco = ACOGraphColoring(
    graph, 
    iterations=5,  # Small number for testing
    ant_count=3,   # Small number for testing
    verbose=True,
    viz_dir=str(viz_dir)
)

print("Running ACO with step-by-step visualization...")
print("(This will be slower due to visualization generation)")
print()
result = aco.run()

print(f"\n✓ ACO completed!")
print(f"  Best color count: {result['color_count']}")
print(f"  Iterations: {result['iterations']}")
print()
print(f"✓ All visualizations saved to: {viz_dir}/")
print(f"\nDirectory structure:")

