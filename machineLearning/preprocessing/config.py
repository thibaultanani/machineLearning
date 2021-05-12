# Fichier de config pour les heuritiques

general = {
    "dataset": "scene",
    "target": "Urban",
    "heuristic": "tabu",
    "deletecol": True,
    "dropna": True,
    "normalize": True,
    "createdummies": False,
    "dropcol": [],
    "dropclass": [],
    "thresholddrop": 70,
    "method": ["svm"],
    "metric": "accuracy",
}

genetic = {
    "pop": 20,
    "gen": 200,
    "mut": 5,
}

differential = {
    "pop": 20,
    "gen": 200,
    "cross proba": 0.5,
    "F": 1,
}

swarm = {
    "pop": 20,
    "gen": 200,
    "w": 0.5,
    "c1": 0.5,
    "c2": 0.5,
}

ant = {
    "pop": 5,
    "gen": 5,
    "p": 0.5,
    "phi": 0.8,
    "alpha": 2
}

hill = {
    "gen": 500,
    "nei": 20,
}

tabu = {
    "tab": 200,
    "gen": 500,
    "nei": 20,
}

simulated = {
    "temperature": 500,
    "alpha": 1,
    "final": 0,
}

vns = {
    "gen": 5,
    "gen vnd": 2,
    "nei": 2,
    "kmax": 2,
}

iterated = {
    "gen": 5,
    "gen vnd": 2,
    "nei": 2,
    "kmax": 2,
}