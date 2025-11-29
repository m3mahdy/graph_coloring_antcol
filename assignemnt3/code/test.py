import networkx as nx

from assignemnt3.code.aco_graph_coloring import ACOGraphColoring


G = nx.generators.classic.turan_graph(30, 3)  # sample graph

aco = ACOGraphColoring(G, num_colors=3)
best_sol, score = aco.run()

print("Conflicts:", score)
