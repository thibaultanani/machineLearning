# Fichier de config pour les heuritiques

general = {
    "dataset": "madelon",
    "target": "Class",
    "heuristic": "differential",
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
    "gen": 10,
    "mut": 5,
}

differential = {
    "pop": 20,
    "gen": 10,
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

proba = {
    "pop": 20,
    "gen": 10,
    "cross proba": 0.5,
    "F": 1,
    "low_bound": 0.1,
    "up_bound": 0.9,
}

pbil = {
    "pop": 20,
    "gen": 10,
    "learning_rate": 0.1,
    "mut_proba": 0.2,
    "mut_shift": 0.05,
}

pbil_diff = {
    "pop": 20,
    "gen": 20,
    "F": 1,
    "learning_rate": 0.1,
    "mut_proba": 0.2,
    "mut_shift": 0.05,
}

hill = {
    "gen": 10,
    "nei": 20,
    "dist": 1,
}

tabu = {
    "tab": 200,
    "gen": 10,
    "nei": 20,
    "dist": 1,
}

simulated = {
    "temperature": 20,
    "alpha": 1,
    "final": 0,
    "dist": 1,
}

random = {
    "gen": 10,
    "nei": 20,
    "p": 1/2,
}