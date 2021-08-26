# Fichier de config pour les heuritiques

general = {
    "dataset": "madelon",
    "target": "Class",
    "heuristic": "vns",
    "deletecol": True,
    "dropna": True,
    "normalize": True,
    "createdummies": False,
    "dropcol": [],
    "dropclass": [],
    "thresholddrop": 70,
    "method": ["lr", "rrc", "knn"],
    "metric": "accuracy",
}

genetic = {
    "pop": 20,
    "gen": 3,
    "mut": 5,
}

differential = {
    "pop": 20,
    "gen": 3,
    "cross proba": 0.5,
    "F": 1,
}

swarm = {
    "pop": 20,
    "gen": 3,
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
    "gen": 10,
    "nei": 20,
}

tabu = {
    "tab": 200,
    "gen": 10,
    "nei": 20,
}

simulated = {
    "temperature": 20,
    "alpha": 1,
    "final": 0,
}
vns = {
    "gen": 20,
    "gen vnd": 2,
    "nei": 2,
    "kmax": 2,
}

iterated = {
    "gen": 20,
    "gen vnd": 2,
    "nei": 5,
    "kmax": 2,
}

random = {
    "gen": 10,
    "nei": 20,
    "p": 1/2,
}