import random
import networkx as nx
import numpy as np

class ACOGraphColoring:
    def __init__(self, graph, num_colors, alpha=1.0, beta=3.0, rho=0.1, ant_count=20, Q=1):
        self.graph = graph
        self.num_colors = num_colors
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ant_count = ant_count
        self.Q = Q

        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)

        # pheromone matrix: node × color
        self.pheromone = np.ones((self.n, self.num_colors))

    def heuristic(self, node, color, solution):
        # simple heuristic: penalize conflicts
        conflicts = sum(1 for neigh in self.graph.neighbors(node)
                        if solution[neigh] == color)
        return 1 / (1 + conflicts)

    def construct_solution(self):
        solution = {node: None for node in self.nodes}

        for node in self.nodes:
            probs = []
            for color in range(self.num_colors):
                tau = self.pheromone[node][color] ** self.alpha
                eta = self.heuristic(node, color, solution) ** self.beta
                probs.append(tau * eta)

            probs = np.array(probs)
            probs = probs / probs.sum()

            chosen_color = np.random.choice(range(self.num_colors), p=probs)
            solution[node] = chosen_color

        return solution

    def evaluate(self, solution):
        conflicts = 0
        for u, v in self.graph.edges():
            if solution[u] == solution[v]:
                conflicts += 1
        return conflicts

    def update_pheromones(self, solutions):
        self.pheromone *= (1 - self.rho)

        for solution, score in solutions:
            if score == 0:
                score = 0.0001  # avoid division by zero

            for node in solution:
                color = solution[node]
                self.pheromone[node][color] += self.Q / score

    def run(self, iterations=50):
        best_solution = None
        best_score = float('inf')

        for _ in range(iterations):
            solutions = []
            for _ in range(self.ant_count):
                sol = self.construct_solution()
                score = self.evaluate(sol)
                solutions.append((sol, score))

                if score < best_score:
                    best_score = score
                    best_solution = sol

            self.update_pheromones(solutions)

            if best_score == 0:  # perfect coloring
                break

        return best_solution, best_score
