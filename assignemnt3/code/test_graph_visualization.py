"""
Test script to verify graph visualization is working correctly.
"""
import sys
from pathlib import Path
import networkx as nx

# Add code path
sys.path.insert(0, str(Path(__file__).parent))

from visualization_utils import save_colored_graph_image

# Create a simple test graph
print("Creating test graph...")
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

# Create a simple solution
solution = {0: 0, 1: 1, 2: 2, 3: 1}

# Test saving
test_path = Path(__file__).parent.parent / 'data' / 'test_graph_viz.png'
print(f"Saving test graph to: {test_path}")

try:
    save_colored_graph_image(
        graph=G,
        solution=solution,
        graph_name="Test Graph",
        color_count=3,
        conflict_count=0,
        save_path=test_path,
        node_size=300
    )
    print(f"✓ Graph saved successfully!")
    print(f"✓ File exists: {test_path.exists()}")
    print(f"✓ File size: {test_path.stat().st_size} bytes")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test multiple saves in sequence
print("\nTesting multiple sequential saves...")
for i in range(3):
    test_path_i = Path(__file__).parent.parent / 'data' / f'test_graph_viz_{i}.png'
    try:
        save_colored_graph_image(
            graph=G,
            solution=solution,
            graph_name=f"Test Graph {i}",
            color_count=3,
            conflict_count=0,
            save_path=test_path_i,
            node_size=300
        )
        print(f"  ✓ Graph {i} saved: {test_path_i.stat().st_size} bytes")
    except Exception as e:
        print(f"  ✗ Graph {i} failed: {e}")

print("\n✓ All tests completed!")
