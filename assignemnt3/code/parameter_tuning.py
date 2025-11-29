import optuna
from assignemnt3.code.aco_graph_coloring import ACOGraphColoring
import networkx as nx



def objective(trial):
    alpha = trial.suggest_float("alpha", 0.1, 3.0)
    beta = trial.suggest_float("beta", 0.1, 5.0)
    rho = trial.suggest_float("rho", 0.01, 0.5)
    ant_count = trial.suggest_int("ant_count", 5, 50)
    Q = trial.suggest_float("Q", 0.1, 5.0)

    aco = ACOGraphColoring(
        graph=G,
        num_colors=3,
        alpha=alpha,
        beta=beta,
        rho=rho,
        ant_count=ant_count,
        Q=Q
    )

    _, score = aco.run(iterations=40)

    trial.set_user_attr("conflicts", score)
    return score
