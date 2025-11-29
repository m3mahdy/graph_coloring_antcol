
Slide 1: Title Slide

Tabu Search for Graph Coloring
A Single-solution based Metaheuristic Approach
---

Slide 2: Presentation Outline

1.  Problem Statement: The Graph Coloring Problem (GCP)
2.  Methodology: Algorithms Employed
3.  Core Algorithm: Tabu Search Mechanism
4.  Experimental Setup: Datasets & Parameters
5.  Results: Comparative Analysis
6.  Discussion & Conclusion

---

Slide 3: Problem Statement: The Graph Coloring Problem (GCP)

Definition: Assign a color to each vertex in a graph $G=(V,E)$.
Core Constraint: Adjacent vertices must have different colors.
    $\forall (u,v) \in E$, $C(u) \ne C(v)$.
Objective: Find the minimum number of colors $k$ (the chromatic number $\chi(G)$).
Complexity: The GCP is NP-hard.

---

Slide 4: Objective Function & Solution

Objective Function: Minimize the conflict count $F(C)$.
    $F(C) = \sum_{(u,v)\in E}\mathbb{I}(C(u)=C(v))$
Feasible Solution: A solution is feasible (a proper coloring) when $F(C)=0$.
Solution Representation: A vector of colors, one for each vertex.
    `Solution = [C(v_1), C(v_2), ..., C(v_n)]`

---

Slide 5: Methodology: Algorithms Employed

1.  Greedy Algorithm (Constructive Heuristic)
    Used to generate a fast, initial baseline solution.
    This initial solution is then fed into the Tabu Search algorithm for refinement.
2.  Tabu Search (S-metaheuristic)
    The core improvement heuristic used in this report.
    It starts with the Greedy solution and iteratively refines it to find a near-optimal coloring.

---

Slide 6: Core Algorithm: Tabu Search Strategy

The implementation uses an Iterative Color Reduction Strategy.

1.  Start: Begin with the $k_0$-coloring from the Greedy algorithm.
2.  Attempt Reduction: Set the target to $k = k_0 - 1$ colors.
3.  Search: Run the TS local search to find a feasible solution (where conflicts $F(C) = 0$).
4.  Repeat:
    If successful: Save the solution and attempt to find a solution with $k-1$ colors.
    If failed: Stop the search.
5.  Result: Return the best feasible solution (minimum $k$) found.

---

Slide 7: Tabu Search: Core Mechanism

TS is an enhanced local search that uses memory to escape local optima.

* Tabu List (Short-Term Memory)
    A FIFO queue that stores recent moves (e.g., `(vertex, old_color)`).
    Purpose: Prevents the search from cycling by temporarily forbidding reverse moves.
* Aspiration Criterion (Dynamic)
    Purpose: Overrides the Tabu List if a forbidden move leads to a new best-ever solution.

---

Slide 8: Tabu Search: Efficiency & Diversification

* Randomized Sampling (Efficiency)
    Instead of checking all neighbors, the algorithm samples $R$ random moves from conflicting vertices.
    It accepts the first-improvement move (first move that reduces conflicts).
* Diversification
    * This is ensured through the combination of:
        1.  The Tabu List
        2.  Randomized Sampling
        3.  Iterative Color Reduction

---

Slide 9: Experimental Setup: Datasets

Datasets: Standardized DIMACS Benchmark Instances were used.
Environment: Python 3.x, NetworkX, NumPy.

---

Slide 10: Experimental Setup: TS Parameters

Optimal parameters were determined empirically through systematic experimentation.

* Optimal Values Used:
    Tabu List Size (L): 10
    Random Sampling Size (R): 50
    Max Iterations per Color (T): 7,000

---

Slide 11: Results: Evaluation Matrix

Performance was evaluated using four key metrics:

1.  Solution Quality (k): The final number of colors used.
2.  Percent Deviation (%): Deviation from the Best-Known Solution (BKS).
3.  Robustness: Standard deviation over 5 independent runs.
4.  Computational Effort: Average run time in seconds.

---

Slide 12: Results: Comparison Table

*Results averaged over 5 independent runs.*


Slide 13: Results: Visual Comparison



Observation: The chart shows Tabu Search (orange) consistently finds a solution with fewer colors than the baseline Greedy algorithm (blue) for all instances.

---

Slide 14: Discussion and Analysis

* Finding 1: Superior Solution Quality
    Tabu Search consistently and significantly outperformed the Greedy algorithm.

* Finding 2: The "Time vs. Quality" Compromise
    This higher quality comes at a massive increase in computational time.
    Greedy: Completed in fractions of a second (e.g., 0.008s - 0.129s).
    Tabu Search: Required up to 3969 seconds (approx. 66 minutes).

---

Slide 15: Limitations and Conclusion

* Limitations
    A considerable gap to the Best Known Solution (BKS) remains.
    This basic TS implementation would need more advanced mechanisms (like long-term memory) to compete with state-of-the-art solvers.
* Conclusion
    The hypothesis is validated: TS finds significantly better solutions than Greedy for the GCP.
    This improvement comes at a high, but predictable, computational cost.

---

Slide 16: Thank You

Questions?