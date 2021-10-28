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
    print("HI")
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


def auto_strategy(F, pop, bestInd, ind_pos, probas):
    strat = ""
    formula = ""
    strat_vector = []

    # Selection des 5 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 5, replace=False)
    xr1, xr2, xr3, xr4, xr5 = pop[selected]

    xr1 = xr1.astype(np.float32)
    xr2 = xr2.astype(np.float32)
    xr3 = xr3.astype(np.float32)
    xr4 = xr4.astype(np.float32)
    xr5 = xr5.astype(np.float32)
    bestInd = bestInd.astype(np.float32)
    current = pop[ind_pos].astype(np.float32)

    # print(xr1)
    # print(type(xr1))

    xr1 = "np.array([" + ",".join(map(str, xr1)) + "])"
    xr2 = "np.array([" + ",".join(map(str, xr2)) + "])"
    xr3 = "np.array([" + ",".join(map(str, xr3)) + "])"
    xr4 = "np.array([" + ",".join(map(str, xr4)) + "])"
    xr5 = "np.array([" + ",".join(map(str, xr5)) + "])"
    bestInd = "np.array([" + ",".join(map(str, bestInd)) + "])"
    current = "np.array([" + ",".join(map(str, current)) + "])"

    if np.random.rand() <= probas[0]:
        # print("individu au hasard")
        strat = strat + xr1 + "+"
        formula = formula + "xr1+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[1]:
        # print("meilleur individu")
        strat = strat + bestInd + "+"
        formula = formula + "xbest+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[2]:
        # print("individu à l'indice i")
        strat = strat + current + "+"
        formula = formula + "xi+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[3]:
        # print("de 1")
        strat = strat + str(F) + "*(" + xr2 + "-" + xr3 + ")+"
        formula = formula + "F*(xr2-xr3)+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[4]:
        # print("de 2")
        strat = strat + str(F) + "*(" + xr4 + "-" + xr5 + ")+"
        formula = formula + "F*(xr4-xr5)+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[5]:
        # print("de rand current")
        strat = strat + str(F) + "*(" + xr1 + "-" + current + ")+"
        formula = formula + "F*(xr1-xi)+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    if np.random.rand() <= probas[6]:
        # print("de best current")
        strat = strat + str(F) + "*(" + bestInd + "-" + current + ")+"
        formula = formula + "F*(xbest-xi)+"
        strat_vector.append(True)
    else:
        strat_vector.append(False)

    strat = strat[:-1]
    formula = formula[:-1]

    if not strat:
        strat = xr1 + "+" + str(F) + "*(" + xr2 + "-" + xr3 + ")"

    return np.clip(eval(strat), 0, 1), formula, strat_vector


if __name__ == '__main__':
    pop = [[True, False, True], [True, True, True], [True, False, False], [False, False, True], [True, True, False],
           [False, False, False], [False, True, True], [False, True, False], [True, True, True], [True, False, True]]
    best = pop[7]
    indice = 0

    probas = [0.5] * 7
    print(pop)
    print(best)
    print(pop[indice])
    print(probas)

    res1, res2, res3 = auto_strategy(1, np.array(pop), np.array(best), indice, probas)
    # print(res1)
    # print(eval(res1))
    # print(np.clip(eval(res1), 0, 1))
    # print(res2)
    # print(res3)