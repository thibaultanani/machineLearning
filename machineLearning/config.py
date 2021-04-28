# Fichier de config pour les heuritiques

general = {
    "dataset": "madelon",
    "target": "Class",
    "heuristic": "genetic",
    "deletecol": True,
    "dropna": True,
    "normalize": False,
    "createdummies": False,
    "dropcol": [],
    "dropclass": [],
    "thresholddrop": 70,
    "method": ["lr"],
    "metric": "accuracy",
}

genetic = {
    "pop": 10,
    "gen": 10,
    "mut": 5,
}

differential = {
    "pop": 5,
    "gen": 5,
    "cross proba": 0.5,
    "F": 1,
}

swarm = {
    "pop": 5,
    "gen": 5,
    "w": 0.5,
    "c1": 0.5,
    "c2": 0.5,
}

hill = {
    "gen": 5,
    "nei": 5,
}

tabu = {
    "tab": 200,
    "gen": 5,
    "nei": 5,
}

simulated = {
    "temperature": 10,
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