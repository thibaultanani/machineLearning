import numpy as np


# Les stratégies de mutations
def de_rand_1(F, pop, bestInd, ind_pos):
    # Selection des 3 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 3, replace=False)
    xr1, xr2, xr3 = pop[selected]

    mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_best_1(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    mutant = bestInd.astype(np.float32) + F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_current_to_rand_1(F, pop, bestInd, ind_pos):
    # Selection des 3 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 3, replace=False)
    xr1, xr2, xr3 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (xr1.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr2.astype(np.float32) - xr3.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_current_to_best_1(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (bestInd.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_rand_2(F, pop, bestInd, ind_pos):
    # Selection des 5 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 5, replace=False)
    xr1, xr2, xr3, xr4, xr5 = pop[selected]

    mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32)) + \
             F * (xr4.astype(np.float32) - xr5.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_best_2(F, pop, bestInd, ind_pos):
    # Selection des 4 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 4, replace=False)
    xr1, xr2, xr3, xr4 = pop[selected]

    mutant = bestInd.astype(np.float32) + F * (xr1.astype(np.float32) - xr2.astype(np.float32)) + \
             F * (xr3.astype(np.float32) - xr4.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_current_to_rand_2(F, pop, bestInd, ind_pos):
    # Selection des 5 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 5, replace=False)
    xr1, xr2, xr3, xr4, xr5 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (xr1.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr2.astype(np.float32) - xr3.astype(np.float32)) + \
             F * (xr4.astype(np.float32) - xr5.astype(np.float32))

    return np.clip(mutant, 0, 1)


def de_current_to_best_2(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 4, replace=False)
    xr1, xr2, xr3, xr4 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (bestInd.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr1.astype(np.float32) - xr2.astype(np.float32)) + \
             F * (xr3.astype(np.float32) - xr4.astype(np.float32))

    return np.clip(mutant, 0, 1)